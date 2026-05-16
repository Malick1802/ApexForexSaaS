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
logger = logging.getLogger("MassAudit")

def run_mass_audit(days=14):
    mt5 = get_mt5()
    
    # 1. Macro Cache
    macro_tickers = {"SP500": "^GSPC", "OIL": "CL=F", "DXY": "DX-Y.NYB", "TNX": "^TNX", "VIX": "^VIX"}
    macro_raw = {}
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: macro_raw[k] = df

    custom_objs = {'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork}
    global_model = tf.keras.models.load_model(str(PROJECT_ROOT / "models" / "foundation_v3" / "foundation_brain.keras"), custom_objects=custom_objs)

    # 2. Find completed specialists
    spec_base = PROJECT_ROOT / "models" / "specialist"
    pairs = [p.name for p in spec_base.iterdir() if (p / "specialist_brain.keras").exists()]
    
    print("\n" + "="*120)
    print(f" MASS FLEET AUDIT: 60% SNIPER ACCURACY (Last {days} Days)")
    print("="*120)
    print(f"{'PAIR':<12} | {'GLOBAL ACC':<12} | {'SPEC ACC':<12} | {'ENSEMBLE ACC':<15} | {'VOL'}")
    print("-" * 120)
    
    for symbol in pairs:
        try:
            # 3. Load Specialist
            spec_path = spec_base / symbol / "specialist_brain.keras"
            spec_model = tf.keras.models.load_model(str(spec_path), custom_objects=custom_objs)
            
            # 4. Data
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
            
            # 5. Predict
            g_preds = global_model.predict(X, verbose=0)
            s_preds = spec_model.predict(X, verbose=0)
            
            g_y = np.argmax(g_preds, axis=1)
            g_conf = np.max(g_preds, axis=1)
            s_y = np.argmax(s_preds, axis=1)
            s_conf = np.max(s_preds, axis=1)
            
            # 6. Calc Stats (60% Threshold)
            t = 0.60
            g_mask = (g_conf >= t) & (g_y != 1)
            s_mask = (s_conf >= t) & (s_y != 1)
            e_mask = (g_y == s_y) & (g_y != 1) & (g_conf >= t) & (s_conf >= t)
            
            g_acc = np.mean(g_y[g_mask] == y_true[g_mask]) if np.any(g_mask) else 0
            s_acc = np.mean(s_y[s_mask] == y_true[s_mask]) if np.any(s_mask) else 0
            e_acc = np.mean(g_y[e_mask] == y_true[e_mask]) if np.any(e_mask) else 0
            
            print(f"{symbol:<12} | {g_acc:<12.1%} | {s_acc:<12.1%} | {e_acc:<15.1%} | {np.sum(e_mask)}")
            
            # Cleanup
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Audit failed for {symbol}: {e}")

    print("="*120)

if __name__ == "__main__":
    run_mass_audit()
