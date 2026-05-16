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
logger = logging.getLogger("Global_Fleet_Audit")

# Full pair list from v3 config
ALL_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "GBPJPY","EURJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","GBPCHF",
    "EURGBP","AUDNZD","NZDCHF","NZDCAD","CADCHF","AUDCHF","EURCAD",
    "GBPNZD","EURNZD","GBPCAD","USDSGD","EURAUD","EURCHF","GBPAUD",
    "AUDCAD", "GOLD"
]

def run_fleet_audit(days=14):
    mt5 = get_mt5()
    if not mt5: return
    
    macro_tickers = {
        "SP500": "^GSPC", "OIL": "CL=F", "NASDAQ": "^IXIC",
        "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX",
        "DXY": "DX-Y.NYB", "COPPER": "HG=F", "BTC": "BTC-USD"
    }
    
    logger.info(f"Fetching {len(ALL_PAIRS)} pairs for fleet audit...")
    raw = {}
    for p in ALL_PAIRS:
        try:
            raw[p] = fetch_mt5_pair(mt5, p, days + 5)
        except: continue
        
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: raw[k] = df

    common = None
    for df in raw.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
    
    logger.info(f"Audit Window: {common[0]} to {common[-1]} ({len(common)} bars)")

    results = []
    custom_objs = {
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork
    }

    for symbol in ALL_PAIRS:
        if symbol not in aligned: continue
        logger.info(f"Auditing {symbol} SPECIALIST...")
        try:
            # 1. Load Specialist Model
            spec_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
            if not spec_path.exists():
                logger.warning(f"  No specialist found for {symbol}, skipping.")
                continue
            model = tf.keras.models.load_model(str(spec_path), custom_objects=custom_objs)
            
            # 2. Build Features
            pair_df = aligned[symbol]
            feat = build_base_features(pair_df)
            feat = add_global_context_v3(feat, aligned, feat.index)
            for col in feat.columns:
                feat[col] = rolling_zscore(feat[col])
            feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Log final feature count for user confirmation
            logger.info(f"  {symbol} Final Features: {feat.shape[1]} (Expect 59)")
            
            labels = triple_barrier_label(pair_df.reindex(feat.index))
            
            X, y_true = [], []
            feat_vals = feat.values.astype(np.float32)
            label_vals = labels.values.astype(np.int32)
            
            for i in range(48, len(feat_vals) - 24):
                X.append(feat_vals[i-48:i])
                y_true.append(label_vals[i])
                
            X = np.array(X)
            y_true = np.array(y_true)
            
            preds = model.predict(X, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            
            active_mask = (y_pred != 1)
            if np.any(active_mask):
                signal_acc = np.mean(y_pred[active_mask] == y_true[active_mask])
                # Filter for high-confidence only
                conf_mask = (np.max(preds, axis=1) > 0.60) & active_mask # SNIPER 60%
                if np.any(conf_mask):
                    conf_acc = np.mean(y_pred[conf_mask] == y_true[conf_mask])
                    conf_count = np.sum(conf_mask)
                else:
                    conf_acc, conf_count = 0, 0
                    
                trades_count = np.sum(active_mask)
            else:
                signal_acc, conf_acc, trades_count, conf_count = 0, 0, 0, 0
                
            results.append({
                "symbol": symbol,
                "raw_acc": signal_acc,
                "conf_acc": conf_acc,
                "trades": trades_count,
                "conf_trades": conf_count
            })
            
            logger.info(f"  {symbol}: Signal Acc: {signal_acc:.1%} | Sniper Acc (60%): {conf_acc:.1%} ({conf_count} trades)")
            
            # Memory Cleanup
            del model, X, y_true, feat, labels
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.warning(f"  {symbol} failed: {e}")
        except Exception as e:
            logger.warning(f"  {symbol} failed: {e}")

    # Final Summary Table
    print("\n" + "="*80)
    print(f"{'SYMBOL':<10} | {'RAW ACC':<10} | {'CONF ACC':<10} | {'TRADES':<10}")
    print("-"*80)
    for r in sorted(results, key=lambda x: x['conf_acc'], reverse=True):
        print(f"{r['symbol']:<10} | {r['raw_acc']:<10.1%} | {r['conf_acc']:<10.1%} | {r['conf_trades']:<10}")
    print("="*80)

if __name__ == "__main__":
    run_fleet_audit(14)
