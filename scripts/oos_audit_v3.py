import os, sys, logging, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mt5_connector import get_mt5
from models.foundation_trainer_v3 import (
    fetch_mt5_pair, fetch_yf_macro, build_base_features, 
    add_global_context_v3, rolling_zscore, triple_barrier_label
)
from models.global_brain import GatedResidualNetwork, VariableSelectionNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OOS_Audit")

def run_oos_audit(days=14):
    # 1. Load Model
    model_path = PROJECT_ROOT / "models" / "foundation_v3" / "foundation_brain.keras"
    custom_objs = {
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork
    }
    logger.info("Loading Global Brain v3 for Audit...")
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objs)
    
    # 2. Fetch Data (14 days + 48h lookback + 24h horizon)
    mt5 = get_mt5()
    if not mt5: return
    
    # Audit a diverse slice of the market
    test_pairs = ["EURUSD", "GOLD", "GBPUSD", "USDJPY", "AUDUSD"]
    macro_tickers = {
        "SP500": "^GSPC", "OIL": "CL=F", "NASDAQ": "^IXIC",
        "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX",
        "DXY": "DX-Y.NYB", "COPPER": "HG=F", "BTC": "BTC-USD"
    }
    
    logger.info(f"Fetching last {days} days for audit...")
    raw = {}
    for p in test_pairs:
        raw[p] = fetch_mt5_pair(mt5, p, days + 5) # Extra for indicators
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: raw[k] = df

    # 3. Align
    common = None
    for df in raw.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
    
    logger.info(f"Audit Window: {common[0]} to {common[-1]} ({len(common)} bars)")

    # 4. Evaluate per pair
    results = []
    
    for symbol in test_pairs:
        logger.info(f"Auditing {symbol}...")
        pair_df = aligned[symbol]
        
        # Features
        feat = build_base_features(pair_df)
        feat = add_global_context_v3(feat, aligned, feat.index)
        for col in feat.columns:
            feat[col] = rolling_zscore(feat[col])
        feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Labels (Truth)
        labels = triple_barrier_label(pair_df.reindex(feat.index))
        
        # Sequence Generation
        X, y_true = [], []
        feat_vals = feat.values.astype(np.float32)
        label_vals = labels.values.astype(np.int32)
        
        # We start from 48h in to have full context
        for i in range(48, len(feat_vals) - 24):
            X.append(feat_vals[i-48:i])
            y_true.append(label_vals[i])
            
        X = np.array(X)
        y_true = np.array(y_true)
        
        # Predict
        preds = model.predict(X, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        # Metrics
        accuracy = np.mean(y_pred == y_true)
        
        # Win Rate on active signals (Ignoring "WAIT" predictions)
        active_mask = (y_pred != 1)
        if np.any(active_mask):
            active_acc = np.mean(y_pred[active_mask] == y_true[active_mask])
            signals_count = np.sum(active_mask)
        else:
            active_acc = 0
            signals_count = 0
            
        results.append({
            "symbol": symbol,
            "accuracy": accuracy,
            "signal_accuracy": active_acc,
            "signals": signals_count
        })
        
        logger.info(f"  {symbol}: Overall Acc: {accuracy:.2%} | Signal Acc: {active_acc:.2%} ({signals_count} trades)")

    # 5. Final Report
    logger.info("="*60)
    logger.info("FINAL OOS AUDIT REPORT (LAST 14 DAYS)")
    logger.info("="*60)
    for r in results:
        logger.info(f"{r['symbol']:8} | Total Acc: {r['accuracy']:.1%} | Trade Acc: {r['signal_accuracy']:.1%} | Trades: {r['signals']}")
    
    avg_trade_acc = np.mean([r['signal_accuracy'] for r in results if r['signals'] > 0])
    logger.info("="*60)
    logger.info(f"AVERAGE OOS TRADE ACCURACY: {avg_trade_acc:.2%}")
    logger.info("="*60)

if __name__ == "__main__":
    run_oos_audit(14)
