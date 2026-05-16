import os, sys, logging, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mt5_connector import get_mt5
from models.foundation_trainer_v3 import (
    fetch_mt5_pair, fetch_yf_macro, build_base_features, 
    add_global_context_v3, rolling_zscore, triple_barrier_label
)
from models.global_brain import GatedResidualNetwork, VariableSelectionNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChampionAudit")

def run_champion_audit(days=14):
    mt5 = get_mt5()
    
    # Pre-fetch Macro
    macro_tickers = {
        "SP500": "^GSPC", "OIL": "CL=F", "NASDAQ": "^IXIC",
        "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX",
        "DXY": "DX-Y.NYB", "COPPER": "HG=F", "BTC": "BTC-USD"
    }
    macro_raw = {}
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: macro_raw[k] = df

    custom_objs = {'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork}
    
    # 1. Champion Map (Based on our Audits)
    champions = {
        "GOLD": {"type": "specialist", "threshold": 0.65},
        "GBPUSD": {"type": "global", "threshold": 0.60},
        "AUDUSD": {"type": "specialist", "threshold": 0.55},
        "EURUSD": {"type": "specialist", "threshold": 0.60},
        "USDJPY": {"type": "global", "threshold": 0.52}
    }
    
    # Load Global Brain once
    global_model = tf.keras.models.load_model(str(PROJECT_ROOT / "models" / "foundation_v3" / "foundation_brain.keras"), custom_objects=custom_objs)

    print("\n" + "="*80)
    print(f" CHAMPION ROUTER AUDIT (Last {days} Days)")
    print("="*80)
    print(f"{'PAIR':<10} | {'CHAMPION':<12} | {'THRESH':<8} | {'ACC':<8} | {'VOL'}")
    print("-" * 80)
    
    total_correct = 0
    total_trades = 0

    for symbol, config in champions.items():
        try:
            # 2. Load the Correct Model
            if config["type"] == "specialist":
                model_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
                if not model_path.exists(): continue
                model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objs)
            else:
                model = global_model
            
            # 3. Fetch & Process
            raw = {symbol: fetch_mt5_pair(mt5, symbol, days + 5)}
            raw.update(macro_raw)
            common = None
            for df in raw.values():
                common = df.index if common is None else common.intersection(df.index)
            aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
            
            feat = build_base_features(aligned[symbol])
            feat = add_global_context_v3(feat, aligned, feat.index)
            for col in feat.columns: feat[col] = rolling_zscore(feat[col])
            feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
            labels = triple_barrier_label(aligned[symbol].reindex(feat.index))
            
            X, y_true = [], []
            feat_vals = feat.values.astype(np.float32)
            label_vals = labels.values.astype(np.int32)
            for i in range(48, len(feat_vals) - 24):
                X.append(feat_vals[i-48:i])
                y_true.append(label_vals[i])
            X = np.array(X); y_true = np.array(y_true)
            
            # 4. Predict & Threshold
            preds = model.predict(X, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            confidences = np.max(preds, axis=1)
            
            mask = (confidences >= config["threshold"]) & (y_pred != 1)
            if np.any(mask):
                correct = np.sum(y_pred[mask] == y_true[mask])
                count = np.sum(mask)
                acc = correct / count
                print(f"{symbol:<10} | {config['type']:<12} | {config['threshold']:<8.0%} | {acc:<8.1%} | {count}")
                
                total_correct += correct
                total_trades += count
            else:
                print(f"{symbol:<10} | {config['type']:<12} | {config['threshold']:<8.0%} | {'N/A':<8} | 0")
            
            # Cleanup
            if config["type"] == "specialist":
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")

    print("-" * 80)
    if total_trades > 0:
        final_acc = total_correct / total_trades
        print(f"{'TOTAL FLEET':<10} | {'-':<12} | {'-':<8} | {final_acc:<8.1%} | {total_trades}")
    print("="*80)

if __name__ == "__main__":
    run_champion_audit()
