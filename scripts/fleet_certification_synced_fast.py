import sys
import os
import logging
import json
import sqlite3
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import InferenceEngine
from core.database import SignalDatabase
from core.performance_gate import get_performance_gate
from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from data_pipeline.labeling import triple_barrier_label

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - FAST_AUDIT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "certification_audit_fast.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FastAudit")

def run_fast_audit():
    logger.info("🚀 Starting Accelerated 14-Day Fleet Audit")
    
    # 1. Init
    db = SignalDatabase()
    conn = sqlite3.connect(str(PROJECT_ROOT / "signals.db"))
    conn.execute("DELETE FROM signals")
    conn.commit()
    conn.close()
    logger.info("🗑️ Database purged.")

    data_engine = DataEngine()
    feature_engineer = FeatureEngineer()
    global_engineer = GlobalFeatureEngineer()
    engine = InferenceEngine() # Use production engine for model loading/routing
    
    symbols = data_engine.get_all_pairs()
    
    # Simulation window: Last 14 days
    now_utc = pd.Timestamp.now(tz='UTC')
    start_sim = now_utc - timedelta(days=14)
    
    # Pre-fetch global macro data for enrichment
    logger.info("🌍 Refreshing Global Market Matrix...")
    global_symbols = ["GOLD", "BTC-USD", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDJPY"]
    global_data = {}
    for gs in global_symbols:
        gdf = data_engine.fetch(gs, interval="1h", days=60)
        if gdf is not None: global_data[gs] = gdf

    for symbol in symbols:
        logger.info(f"--- Preparing {symbol} ---")
        try:
            # 2. Fetch and Enrich Features (ONCE)
            df = data_engine.fetch(symbol, interval="1h", days=60)
            if df is None or len(df) < 100: continue
            
            df_labeled = triple_barrier_label(df, symbol=symbol)
            base_features = feature_engineer.extract_features(df)
            features = global_engineer.add_global_features(symbol, base_features, global_data)
            
            # 3. Model Setup (Routing)
            models = None
            predators = {p['symbol']: p for p in engine.config.get('fleet', {}).get('predators', [])}
            threshold = 0.60
            
            if symbol in predators:
                route = predators[symbol]['route']
                threshold = predators[symbol].get('threshold', 0.60)
                if route == "global": models = engine.load_foundation_model(symbol)
                elif route in ["specialist", "ensemble"]: models = engine.load_models(symbol)
            
            if not models: models = engine.load_foundation_model(symbol)
            if not models: models = engine.load_models(symbol)
            if not models: continue

            model_type = models.get('model_type', 'unknown')
            version = models.get('version', 'v1')
            scaler = models.get('scaler')
            expected_features = models.get('n_features', 57)
            
            # 4. Simulation Loop
            sim_indices = features.index[features.index >= start_sim]
            signals_count = 0
            active_trade_until = None
            
            logger.info(f"⚡ Simulating {len(sim_indices)} periods for {symbol} ({model_type})...")
            
            for ts in sim_indices:
                # Check for active trade lock
                if active_trade_until and ts < active_trade_until:
                    continue
                
                # Slice features for this 'now'
                seq_len = 48 if version == 'v3' else 60
                feat_slice = features[features.index <= ts].tail(seq_len)
                if len(feat_slice) < seq_len: continue
                
                # 5. Core Prediction logic
                X_input = feat_slice.values.reshape(1, seq_len, -1)
                
                # Feature Alignment/Scaling
                if version == 'v3':
                    history = features[features.index <= ts].tail(720)
                    mu = history.mean()
                    std = history.std().replace(0, 1e-8)
                    norm_slice = (feat_slice - mu) / std
                    X_final = norm_slice.values.reshape(1, seq_len, -1)
                elif scaler:
                    X_flat = X_input.reshape(-1, X_input.shape[2])
                    if X_flat.shape[1] > expected_features: X_flat = X_flat[:, :expected_features]
                    X_final = scaler.transform(X_flat).reshape(1, seq_len, -1)
                else:
                    X_final = X_input

                # Model Call
                try:
                    proba = models['model'].predict(X_final, verbose=0)[0]
                    buy_prob = float(proba[1])
                    sell_prob = float(proba[2])
                except: continue

                signal_type = "WAIT"
                confidence = 0.0
                
                if buy_prob >= threshold and buy_prob > sell_prob:
                    signal_type, confidence = "BUY", buy_prob
                elif sell_prob >= threshold and sell_prob > buy_prob:
                    signal_type, confidence = "SELL", sell_prob
                
                if signal_type != "WAIT":
                    # Determine Outcome
                    truth = df_labeled.loc[ts, 'label']
                    outcome = "FAIL"
                    if signal_type == "BUY" and truth == 1: outcome = "SUCCESS"
                    elif signal_type == "SELL" and truth == 2: outcome = "SUCCESS"
                    elif truth == 0: outcome = "PENDING"
                    
                    if outcome != "PENDING":
                        # Save to DB
                        sig_data = {
                            'symbol': symbol,
                            'timestamp': ts.isoformat(),
                            'signal': signal_type,
                            'confidence': confidence,
                            'buy_prob': buy_prob,
                            'sell_prob': sell_prob,
                            'wait_prob': float(proba[0]),
                            'price_at_signal': float(df.loc[ts, 'close']),
                            'outcome': outcome,
                            'model_type': model_type,
                            'winning_tier': "60%",
                            'is_proven': 0
                        }
                        db.save_signal(sig_data)
                        signals_count += 1
                        
                        # Set dynamic trade lock (until TP/SL/Timeout hit)
                        outcome_bars = int(df_labeled.loc[ts, 'bars_to_outcome'])
                        active_trade_until = ts + timedelta(hours=outcome_bars)

            
            logger.info(f"✅ {symbol}: Recorded {signals_count} signals.")
            
        except Exception as e:
            logger.error(f"Error auditing {symbol}: {e}")

    # 6. Re-Certification
    logger.info("⚖️ Recalculating Performance Matrix...")
    gate = get_performance_gate()
    gate.recompute_from_db(lookback_days=14)
    
    logger.info("✨ Fleet Synchronization Complete.")

if __name__ == "__main__":
    run_fast_audit()
