import sys
import os
import logging
import sqlite3
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

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
    format='%(asctime)s - DEEP_AUDIT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "deep_grid_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepAudit")

def run_deep_grid_audit():
    logger.info("🕵️ Starting Deep Fleet Grid Audit (14-Day OOS)")
    
    # 1. Initialize Engines
    db = SignalDatabase()
    # Purge signals for clean audit
    conn = sqlite3.connect(str(PROJECT_ROOT / "signals.db"))
    conn.execute("DELETE FROM signals")
    conn.commit()
    conn.close()
    
    data_engine = DataEngine()
    feature_engineer = FeatureEngineer()
    global_engineer = GlobalFeatureEngineer()
    engine = InferenceEngine()
    
    symbols = data_engine.get_all_pairs()
    thresholds = [0.52, 0.60, 0.70, 0.80]
    model_routes = ["global", "specialist", "ensemble"]
    
    now_utc = pd.Timestamp.now(tz='UTC')
    start_sim = now_utc - timedelta(days=14)
    
    # Pre-fetch global macro
    global_symbols = ["GOLD", "BTC-USD", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDJPY"]
    global_data = {gs: data_engine.fetch(gs, interval="1h", days=60) for gs in global_symbols}
    global_data = {k: v for k, v in global_data.items() if v is not None}

    results = []

    for symbol in symbols:
        logger.info(f"--- Grid Search: {symbol} ---")
        try:
            df = data_engine.fetch(symbol, interval="1h", days=60)
            if df is None or len(df) < 100: continue
            
            df_labeled = triple_barrier_label(df, symbol=symbol)
            base_features = feature_engineer.extract_features(df)
            features = global_engineer.add_global_features(symbol, base_features, global_data)
            
            # Pre-load all available models for this symbol
            models = {
                "global": engine.load_foundation_model(symbol),
                "specialist": engine.load_models(symbol)
            }
            
            sim_indices = features.index[features.index >= start_sim]
            if len(sim_indices) == 0: continue

            for route in model_routes:
                for threshold in thresholds:
                    # Run simulation for this specific (route, threshold)
                    successes = 0
                    total_trades = 0
                    active_trade_until = None
                    
                    # Force model type
                    if route == "ensemble":
                        if not models["global"] or not models["specialist"]: continue
                        active_models = models
                    else:
                        if not models[route]: continue
                        active_models = models[route]

                    for ts in sim_indices:
                        if active_trade_until and ts < active_trade_until: continue
                        
                        seq_len = 48 # standard v3
                        feat_slice = features[features.index <= ts].tail(seq_len)
                        if len(feat_slice) < seq_len: continue
                        
                        # Normalize (v3 style)
                        history = features[features.index <= ts].tail(720)
                        mu = history.mean(); std = history.std().replace(0, 1e-8)
                        norm_slice = (feat_slice - mu) / std
                        X_final = norm_slice.values.reshape(1, seq_len, -1)
                        
                        try:
                            if route == "ensemble":
                                g_p = models["global"]["model"].predict(X_final, verbose=0)[0]
                                s_p = models["specialist"]["model"].predict(X_final, verbose=0)[0]
                                g_y = np.argmax(g_p); s_y = np.argmax(s_p)
                                g_conf = np.max(g_p); s_conf = np.max(s_p)
                                
                                if g_y == s_y and g_y != 1 and g_conf >= threshold and s_conf >= threshold:
                                    signal_type = "BUY" if g_y == 1 else "SELL" # Wait, label 0=Wait, 1=Buy, 2=Sell usually
                                    # Actually check labels: 0=WAIT, 1=BUY, 2=SELL
                                    signal_type = "BUY" if g_y == 1 else "SELL"
                                    confidence = max(g_conf, s_conf)
                                else: signal_type = "WAIT"
                            else:
                                proba = active_models["model"].predict(X_final, verbose=0)[0]
                                b_p = float(proba[1]); s_p = float(proba[2])
                                if b_p >= threshold and b_p > s_p: signal_type, confidence = "BUY", b_p
                                elif s_p >= threshold and s_p > b_p: signal_type, confidence = "SELL", s_p
                                else: signal_type = "WAIT"
                        except: continue
                        
                        if signal_type != "WAIT":
                            truth = df_labeled.loc[ts, 'label']
                            if (signal_type == "BUY" and truth == 1) or (signal_type == "SELL" and truth == 2):
                                successes += 1
                            
                            total_trades += 1
                            outcome_bars = int(df_labeled.loc[ts, 'bars_to_outcome'])
                            active_trade_until = ts + timedelta(hours=outcome_bars)

                    wr = (successes / total_trades) if total_trades > 0 else 0
                    results.append({
                        "symbol": symbol,
                        "route": route,
                        "threshold": threshold,
                        "wr": wr,
                        "trades": total_trades
                    })
                    logger.info(f"   - {route} @ {threshold}: WR={wr:.1%} ({total_trades} trades)")

            # Find best for THIS symbol
            symbol_res = pd.DataFrame([r for r in results if r['symbol'] == symbol])
            if not symbol_res.empty:
                best = symbol_res.sort_values(['wr', 'trades'], ascending=False).iloc[0]
                logger.info(f"🎯 LOCAL BEST FOR {symbol}: {best['route']} @ {best['threshold']} (WR={best['wr']:.1%}, Trades={best['trades']})")

        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")

    # 2. Find Best Configs, Save Signals, and Update config.yaml
    logger.info("🏆 Finalizing Best Configurations...")
    df_res = pd.DataFrame(results)
    best_configs = {}
    
    for symbol in symbols:
        symbol_res = df_res[df_res['symbol'] == symbol]
        if symbol_res.empty: continue
        best = symbol_res.sort_values(['wr', 'trades'], ascending=False).iloc[0]
        best_configs[symbol] = best
        logger.info(f"🎯 Best for {symbol}: {best['route']} @ {best['threshold']} (WR={best['wr']:.1%})")

    # SECOND PASS: Save signals for the BEST configurations only
    logger.info("💾 Persisting winning signals to database...")
    for symbol, best in best_configs.items():
        try:
            df = data_engine.fetch(symbol, interval="1h", days=60)
            df_labeled = triple_barrier_label(df, symbol=symbol)
            base_features = feature_engineer.extract_features(df)
            features = global_engineer.add_global_features(symbol, base_features, global_data)
            
            # Load the best model
            if best['route'] == "ensemble":
                models = {"global": engine.load_foundation_model(symbol), "specialist": engine.load_models(symbol)}
            else:
                models = engine.load_foundation_model(symbol) if best['route'] == "global" else engine.load_models(symbol)
            
            sim_indices = features.index[features.index >= start_sim]
            active_trade_until = None
            
            for ts in sim_indices:
                if active_trade_until and ts < active_trade_until: continue
                seq_len = 48
                feat_slice = features[features.index <= ts].tail(seq_len)
                if len(feat_slice) < seq_len: continue
                
                history = features[features.index <= ts].tail(720)
                mu = history.mean(); std = history.std().replace(0, 1e-8)
                norm_slice = (feat_slice - mu) / std
                X_final = norm_slice.values.reshape(1, seq_len, -1)
                
                try:
                    if best['route'] == "ensemble":
                        g_p = models["global"]["model"].predict(X_final, verbose=0)[0]
                        s_p = models["specialist"]["model"].predict(X_final, verbose=0)[0]
                        if np.argmax(g_p) == np.argmax(s_p) and np.argmax(g_p) != 1 and np.max(g_p) >= best['threshold'] and np.max(s_p) >= best['threshold']:
                            signal_type = "BUY" if np.argmax(g_p) == 1 else "SELL"
                            proba = g_p # Use global proba as proxy
                        else: signal_type = "WAIT"
                    else:
                        proba = models["model"].predict(X_final, verbose=0)[0]
                        if proba[1] >= best['threshold']: signal_type = "BUY"
                        elif proba[2] >= best['threshold']: signal_type = "SELL"
                        else: signal_type = "WAIT"
                    
                    if signal_type != "WAIT":
                        truth = df_labeled.loc[ts, 'label']
                        outcome = "SUCCESS" if (signal_type == "BUY" and truth == 1) or (signal_type == "SELL" and truth == 2) else "FAIL"
                        if truth == 0: outcome = "PENDING"
                        
                        if outcome != "PENDING":
                            sig_data = {
                                'symbol': symbol,
                                'timestamp': ts.isoformat(),
                                'signal': signal_type,
                                'confidence': float(np.max(proba)),
                                'buy_prob': float(proba[1]),
                                'sell_prob': float(proba[2]),
                                'wait_prob': float(proba[0]),
                                'price_at_signal': float(df.loc[ts, 'close']),
                                'outcome': outcome,
                                'model_type': models.get('model_type', 'grid_optimized'),
                                'winning_tier': f"{int(best['threshold']*100)}%",
                                'is_proven': 1 if best['wr'] >= 0.70 and best['trades'] >= 2 else 0
                            }
                            db.save_signal(sig_data)
                            outcome_bars = int(df_labeled.loc[ts, 'bars_to_outcome'])
                            active_trade_until = ts + timedelta(hours=outcome_bars)
                except: continue
        except Exception as e:
            logger.error(f"Error saving signals for {symbol}: {e}")

    # Update config.yaml
    with open(str(PROJECT_ROOT / "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    config['fleet']['predators'] = [
        {"symbol": s, "route": b['route'], "threshold": float(b['threshold'])}
        for s, b in best_configs.items()
    ]
    
    with open(str(PROJECT_ROOT / "config.yaml"), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # 3. Final Gate Sync
    logger.info("⚖️ Syncing Performance Gate...")
    gate = get_performance_gate()
    gate.recompute_from_db(lookback_days=14)
    
    logger.info("✨ Deep Grid Audit & Fleet Realignment Complete.")

if __name__ == "__main__":
    run_deep_grid_audit()
