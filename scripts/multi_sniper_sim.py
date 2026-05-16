import os, sys, logging
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
logger = logging.getLogger("MultiSniper")

def run_multi_sniper(pairs=["GOLD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"], days=14):
    mt5 = get_mt5()
    
    # Pre-fetch Macro
    macro_tickers = {"DXY": "DX-Y.NYB", "TNX": "^TNX", "VIX": "^VIX", "SP500": "^GSPC"}
    macro_raw = {}
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: macro_raw[k] = df

    thresholds = [0.52, 0.55, 0.60, 0.65]
    
    print("\n" + "="*85)
    print(f" MULTI-PAIR SNIPER SIMULATION (Last {days} Days)")
    print("="*85)
    header = f"{'PAIR':<10} | " + " | ".join([f"{t:<12.0%} " for t in thresholds])
    print(header)
    print("-" * 85)
    
    custom_objs = {'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork}

    for symbol in pairs:
        try:
            model_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
            if not model_path.exists(): continue
            
            model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objs)
            
            # Fetch & Align
            raw = {symbol: fetch_mt5_pair(mt5, symbol, days + 5)}
            raw.update(macro_raw)
            common = None
            for df in raw.values():
                common = df.index if common is None else common.intersection(df.index)
            aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
            
            # Features
            feat = build_base_features(aligned[symbol])
            feat = add_global_context_v3(feat, aligned, feat.index)
            for col in feat.columns: feat[col] = rolling_zscore(feat[col])
            feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
            labels = triple_barrier_label(aligned[symbol].reindex(feat.index))
            
            # Predict
            X, y_true = [], []
            feat_vals = feat.values.astype(np.float32)
            label_vals = labels.values.astype(np.int32)
            for i in range(48, len(feat_vals) - 24):
                X.append(feat_vals[i-48:i])
                y_true.append(label_vals[i])
            X = np.array(X); y_true = np.array(y_true)
            
            preds = model.predict(X, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            confidences = np.max(preds, axis=1)
            
            row = f"{symbol:<10} | "
            for t in thresholds:
                mask = (confidences >= t) & (y_pred != 1)
                if np.any(mask):
                    wr = np.mean(y_pred[mask] == y_true[mask])
                    count = np.sum(mask)
                    row += f"{wr:<5.0%} ({count:<3})  | "
                else:
                    row += f"{'N/A':<12} | "
            print(row)
            
            # Clean up
            del model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")

    print("="*85)

import gc
if __name__ == "__main__":
    run_multi_sniper()
