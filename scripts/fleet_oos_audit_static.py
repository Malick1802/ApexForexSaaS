import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from data_pipeline.labeling import triple_barrier_label
from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - OOS_AUDIT - %(levelname)s - %(message)s')
logger = logging.getLogger("StaticOOS")

# Configuration
TIERS = [0.60, 0.70, 0.80, 0.90, 1.00]
OOS_START_DATE = "2026-03-31" # Foundation model cutoff

def run_static_oos_audit():
    logger.info("🚀 Starting Static Out-of-Sample Fleet Audit (Simulation Mode)")
    
    data_engine = DataEngine()
    symbols = data_engine.get_all_pairs()
    feature_engineer = FeatureEngineer()
    global_engineer = GlobalFeatureEngineer()
    
    # 1. Load Foundation Brain & Scaler
    scaler_path = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
    brain_path = PROJECT_ROOT / "models" / "foundation" / "foundation_brain.keras"
    
    if not brain_path.exists():
        logger.error(f"Foundation brain not found at {brain_path}")
        return
        
    logger.info(f"Loading Foundation Model from {brain_path}")
    model = keras.models.load_model(
        str(brain_path),
        custom_objects={
            'VariableSelectionNetwork': VariableSelectionNetwork,
            'GatedResidualNetwork': GatedResidualNetwork
        }
    )
    scaler = joblib.load(str(scaler_path))
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    
    # Global context for intelligence matrix
    global_data = {}
    for g in ["GOLD", "^TNX"]:
        try:
            gdf = data_engine.fetch(g, interval="1h", days=60)
            if gdf is not None: global_data[g] = gdf
        except: pass

    fleet_results = {}

    for symbol in symbols:
        logger.info(f"--- Auditing {symbol} ---")
        try:
            # Fetch data (including warmup)
            df = data_engine.fetch(symbol, interval="1h", days=60)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue
                
            # Generate truth labels and Bars-to-Outcome
            df_labeled = triple_barrier_label(df, symbol=symbol)
            base_features = feature_engineer.extract_features(df_labeled)
            features = global_engineer.add_global_features(symbol, base_features, global_data)
            
            y_all = df_labeled['label'].astype(int).values
            bto_all = df_labeled['bars_to_outcome'].astype(int).values
            X, _ = feature_engineer.create_sequences(features, y_all, sequence_length=60)
            
            # Slice for OOS only (Post 3/31)
            oos_cutoff_dt = pd.Timestamp(OOS_START_DATE, tz='UTC')
            # features.index[-len(X):] corresponds to X
            oos_mask = features.index[-len(X):] >= oos_cutoff_dt
            
            X_oos = X[oos_mask]
            y_oos = y_all[-len(X):][oos_mask]
            bto_oos = bto_all[-len(X):][oos_mask]
            
            if len(X_oos) == 0:
                logger.warning(f"No OOS data found for {symbol} after {OOS_START_DATE}")
                continue
                
            # Scaling & Bulk Inference
            X_oos_flat = X_oos.reshape(-1, X_oos.shape[2])
            X_oos_flat = (X_oos_flat - mean) / scale
            X_oos_scaled = X_oos_flat.reshape(len(y_oos), 60, -1)
            
            # (Time Series Sequence of Preds)
            raw_preds = model.predict(X_oos_scaled, verbose=0)
            
            # ─────────────────────────────────────────────────────────────
            # SIMULATION ENGINE: Chronological Loop for Isolation
            # ─────────────────────────────────────────────────────────────
            symbol_tiers = {}
            for t in TIERS:
                active_trade_until = -1
                wins = 0
                losses = 0
                pending = 0
                
                for i in range(len(raw_preds)):
                    if i <= active_trade_until:
                        continue # Strict isolation for THIS tier
                        
                    # Model Consensus (0=Wait, 1=Buy, 2=Sell)
                    pred_class = np.argmax(raw_preds[i])
                    confidence = raw_preds[i][pred_class]
                    
                    if pred_class != 0 and confidence >= t:
                        # 🚀 ENTER TRADE
                        truth = y_oos[i]
                        duration = bto_oos[i]
                        
                        # Check PENDING state
                        # If duration extends past end of available data
                        if i + duration >= len(y_oos):
                            pending += 1
                            active_trade_until = len(y_oos) # Block until end
                            continue

                        # Resolve Outcome
                        if pred_class == 1: # BUY
                            if truth == 1: wins += 1
                            else: losses += 1
                        elif pred_class == 2: # SELL
                            if truth == 2: wins += 1
                            else: losses += 1
                            
                        # Set Isolation Block
                        active_trade_until = i + duration
                
                total_resolved = wins + losses
                accuracy = (wins / total_resolved) if total_resolved > 0 else 0.0
                
                symbol_tiers[int(t*100)] = {
                    "accuracy": accuracy,
                    "trades": total_resolved,
                    "pending": pending
                }
                logger.info(f"  Tier {int(t*100)}%: Acc={accuracy:.1%} | Resolved={total_resolved} | Pending={pending}")
            
            fleet_results[symbol] = symbol_tiers
            
        except Exception as e:
            logger.error(f"Failed to audit {symbol}: {e}", exc_info=True)

    # Generate Final Report
    generate_markdown_report(fleet_results, symbols)
    
    # Save JSON results
    with open(PROJECT_ROOT / "logs" / "fleet_oos_results_static.json", "w") as f:
        json.dump(fleet_results, f, indent=2)

def generate_markdown_report(results, symbols):
    report_path = PROJECT_ROOT / "logs" / "fleet_oos_report_static.md"
    
    with open(report_path, "w") as f:
        f.write("# Static OOS Fleet Audit Report (Full Fleet)\n\n")
        f.write(f"**Audit Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**OOS Boundary:** Post 2026-03-31\n\n")
        
        f.write("## Executive Summary\n")
        f.write("> [!NOTE]\n")
        f.write("> **Simulation Logic**: Once a trade is entered, subsequent signals for that pair/tier are ignored until the trade hits TP or SL. Trades ending past today's date are marked as PENDING and excluded from accuracy.\n\n")
        
        # Calc aggregate stats at 70% hurdle
        all_resolved = 0
        all_wins = 0
        all_pending = 0
        for symbol, tiers in results.items():
            t70 = tiers.get(70, {})
            all_resolved += t70.get("trades", 0)
            all_wins += t70.get("accuracy", 0) * t70.get("trades", 0)
            all_pending += t70.get("pending", 0)
            
        avg_acc = (all_wins / all_resolved) if all_resolved > 0 else 0
        f.write(f"* **Total Resolved Trades (70%+):** {all_resolved}\n")
        f.write(f"* **Total Pending Trades (Active):** {all_pending}\n")
        f.write(f"* **Fleet-wide Hurdle Accuracy (70%+):** {avg_acc:.1%}\n\n")
        
        f.write("## Per-Pair Isolated Performance\n\n")
        header = "| Symbol | 60% Acc (Trades) | 70% Acc (Trades) | 80% Acc (Trades) | 90% Acc (Trades) | 100% Acc (Trades) |\n"
        separator = "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        f.write(header)
        f.write(separator)
        
        for symbol in symbols:
            if symbol not in results:
                f.write(f"| {symbol} | N/A | N/A | N/A | N/A | N/A |\n")
                continue
            
            row = f"| **{symbol}** "
            for t in [60, 70, 80, 90, 100]:
                data = results[symbol].get(t, {"accuracy": 0, "trades": 0, "pending": 0})
                row += f"| {data['accuracy']:.1%} ({data['trades']}) [+{data['pending']}nd] "
            row += "|\n"
            f.write(row)
            
        f.write("\n\n> [!TIP]\n")
        f.write("> **Performance Tip**: Pairs with low 'Resolved' counts but high 'Pending' counts are currently in long-duration trades that haven't hit objectives yet.")

    logger.info(f"✅ Isolated Report generated at {report_path}")

if __name__ == "__main__":
    run_static_oos_audit()
