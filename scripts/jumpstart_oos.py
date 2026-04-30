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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - JUMPSTART - %(levelname)s - %(message)s')
logger = logging.getLogger("Jumpstart")

# Configuration
TIERS = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
OOS_START_DATE = "2026-03-16" # 45-day Jumpstart Window

def run_jumpstart_simulation():
    logger.info("🚀 Starting 14-day OOS Jumpstart Simulation (Institutional Parity)")
    
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
    
    # Global context for intelligence matrix
    global_data = {}
    g7_pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "GOLD", "^TNX", "^IRX"]
    for g in g7_pairs:
        try:
            gdf = data_engine.fetch(g, interval="1h", days=60)
            if gdf is not None: global_data[g] = gdf
        except: pass

    fleet_results = {}

    for symbol in symbols:
        logger.info(f"--- Simulating {symbol} ---")
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
            
            # ── Feature Alignment Layer ───────────
            f_cols = [
                'open_norm', 'high_norm', 'low_norm', 'hl_range', 'oc_range',
                'close_ret_1', 'close_ret_5', 'close_ret_10', 'rsi', 'atr_norm',
                'bb_position', 'bb_width_norm', 'macd_norm', 'macd_signal_norm',
                'macd_hist_norm', 'volume_rel', 'volume_ret', 'hour_sin', 'hour_cos',
                'dow_sin', 'dow_cos', 'USD_strength', 'EUR_strength', 'GBP_strength',
                'JPY_strength', 'AUD_strength', 'CAD_strength', 'CHF_strength',
                'NZD_strength', 'dxy_proxy', 'dxy_ret', 'gold_ret', 'vix_proxy',
                'yield_curve_slope'
            ]
            
            # Map pipeline features to expected naming convention
            mapping = {
                'atr': 'atr_norm', 'bb_width': 'bb_width_norm',
                'macd': 'macd_norm', 'macd_signal': 'macd_signal_norm',
                'macd_hist': 'macd_hist_norm', 'volume_norm': 'volume_rel'
            }
            for src, dst in mapping.items():
                if src in features.columns:
                    features[dst] = features[src]
            
            # Close Return Parity
            for i in [1, 5, 10]:
                features[f'close_ret_{i}'] = df['close'].pct_change(i).fillna(0)
            
            # Calculate Time Features (Cyclical)
            ts = features.index
            features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24.0)
            features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24.0)
            features['dow_sin'] = np.sin(2 * np.pi * ts.weekday / 7.0)
            features['dow_cos'] = np.cos(2 * np.pi * ts.weekday / 7.0)
            
            # Always recompute volume_ret from volume_rel for pipeline consistency
            features['volume_ret'] = features['volume_rel'].pct_change().fillna(0)
            
            # Explicit selection and order
            final_features = []
            for c in f_cols:
                if c not in features.columns:
                    features[c] = 0.0 # Zero-fill context
                final_features.append(c)
            
            features = features[final_features]
            y_all_series = df_labeled['label'].astype(int)
            y_all = y_all_series.values
            bto_all = df_labeled['bars_to_outcome'].astype(int).values
            X, _ = feature_engineer.create_sequences(features, y_all_series, sequence_length=60)
            
            # Slice for OOS only
            oos_cutoff_dt = pd.Timestamp(OOS_START_DATE, tz='UTC')
            oos_mask = features.index[-len(X):] >= oos_cutoff_dt
            
            X_oos = X[oos_mask]
            y_oos = y_all[-len(X):][oos_mask]
            bto_oos = bto_all[-len(X):][oos_mask]
            
            if len(X_oos) == 0:
                logger.warning(f"No OOS data found for {symbol} after {OOS_START_DATE}")
                continue
                
            # Scaling & Bulk Inference — use scaler.transform() for correctness
            X_oos_flat = X_oos.reshape(-1, X_oos.shape[2])
            X_oos_flat = scaler.transform(X_oos_flat)
            X_oos_scaled = X_oos_flat.reshape(len(y_oos), 60, -1)
            
            raw_preds = model.predict(X_oos_scaled, verbose=0)
            
            # Simulation Engine
            symbol_tiers = {"BUY": {}, "SELL": {}}
            for direction_idx, direction_name in [(1, 'BUY'), (2, 'SELL')]:
                for t in TIERS:
                    active_trade_until = -1
                    wins = 0
                    losses = 0
                    pending = 0
                    for i in range(len(raw_preds)):
                        if i <= active_trade_until: continue
                        pred_class = np.argmax(raw_preds[i])
                        confidence = raw_preds[i][pred_class]
                        
                        if pred_class == direction_idx and confidence >= t:
                            truth = y_oos[i]
                            duration = bto_oos[i]
                            if i + duration >= len(y_oos):
                                pending += 1
                                active_trade_until = len(y_oos)
                                continue
                            if truth == direction_idx: wins += 1
                            else: losses += 1
                            active_trade_until = i + duration
                    
                    total_resolved = wins + losses
                    accuracy = (wins / total_resolved) if total_resolved > 0 else 0.0
                    symbol_tiers[direction_name][int(t*100)] = {
                        "accuracy": accuracy,
                        "trades": total_resolved,
                        "pending": pending
                    }
            
            fleet_results[symbol] = symbol_tiers
            
        except Exception as e:
            logger.error(f"Failed to simulate {symbol}: {e}")

    # ── WHITELIST JUMPSTART INTEGRATION ────────────────────────────
    update_trading_whitelist(fleet_results)

def update_trading_whitelist(results):
    """Update the trading whitelist with OOS performance."""
    whitelist_path = PROJECT_ROOT / "config" / "trading_whitelist.json"
    if not whitelist_path.exists():
        logger.warning("Whitelist file not found.")
        return
        
    try:
        with open(whitelist_path, 'r') as f:
            whitelist = json.load(f)
            
        updated_count = 0
        for symbol, tiers in results.items():
            if symbol not in whitelist["performance_matrix"]:
                whitelist["performance_matrix"][symbol] = {"BUY": {}, "SELL": {}, "ALL": {}}
            
            for side in ["BUY", "SELL"]:
                for tier_pct, data in tiers.get(side, {}).items():
                    if data['trades'] >= 1: # Jumpstart even with 1 trade
                        approved = data['accuracy'] >= 0.60 and data['trades'] >= 3
                        whitelist["performance_matrix"][symbol][side][str(tier_pct)] = {
                            "win_rate": data['accuracy'],
                            "accuracy": data['accuracy'],
                            "trades": data['trades'],
                            "oos_trades": data['trades'],
                            "oos_accuracy": data['accuracy'],
                            "status": "APPROVED" if approved else "BENCHED",
                            "last_updated": datetime.now().isoformat(),
                            "source": "OOS Jumpstart (45d)"
                        }
                        updated_count += 1
        
        whitelist['last_updated'] = datetime.now().isoformat()
        with open(whitelist_path, 'w') as f:
            json.dump(whitelist, f, indent=2)
            
        logger.info(f"✅ Whitelist Jumpstart Complete: Updated {updated_count} records.")
    except Exception as e:
        logger.error(f"Failed to update whitelist: {e}")

if __name__ == "__main__":
    run_jumpstart_simulation()
