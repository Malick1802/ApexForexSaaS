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
logger = logging.getLogger("GodTable")

def run_ultimate_audit(days=14, cooldown=24):
    mt5 = get_mt5()
    
    # 1. Macro Cache
    macro_tickers = {"SP500": "^GSPC", "OIL": "CL=F", "DXY": "DX-Y.NYB", "TNX": "^TNX", "VIX": "^VIX"}
    macro_raw = {}
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: macro_raw[k] = df

    custom_objs = {'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork}
    global_model = tf.keras.models.load_model(str(PROJECT_ROOT / "models" / "foundation_v3" / "foundation_brain.keras"), custom_objects=custom_objs)

    spec_base = PROJECT_ROOT / "models" / "specialist"
    pairs = [p.name for p in spec_base.iterdir() if (p / "specialist_brain.keras").exists()]
    thresholds = [0.52, 0.60, 0.70]
    
    print("\n" + "="*140)
    print(f" THE GOD TABLE: GLOBAL vs SPECIALIST vs ENSEMBLE (SPARSE) (Last {days} Days)")
    print("="*140)
    print(f"{'PAIR':<10} | {'THRESH':<6} | {'GLOBAL ACC':<12} | {'SPEC ACC':<12} | {'ENS ACC':<12} | {'ENS VOL'}")
    print("-" * 140)
    
    for symbol in pairs:
        try:
            # 2. Load Specialist
            spec_path = spec_base / symbol / "specialist_brain.keras"
            spec_model = tf.keras.models.load_model(str(spec_path), custom_objects=custom_objs)
            
            # 3. Data
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
            
            # 4. All Predictions
            g_preds = global_model.predict(X, verbose=0)
            s_preds = spec_model.predict(X, verbose=0)
            g_y, g_conf = np.argmax(g_preds, axis=1), np.max(g_preds, axis=1)
            s_y, s_conf = np.argmax(s_preds, axis=1), np.max(s_preds, axis=1)
            
            # 5. Sparse Eval for each threshold
            for t in thresholds:
                # Simulation Logic
                last_g, last_s, last_e = -cooldown, -cooldown, -cooldown
                g_trades, s_trades, e_trades = [], [], []
                
                for i in range(len(X)):
                    # Global Solo
                    if i >= last_g + cooldown:
                        if (g_y[i] != 1) and (g_conf[i] >= t):
                            g_trades.append(g_y[i] == y_true[i])
                            last_g = i
                    # Spec Solo
                    if i >= last_s + cooldown:
                        if (s_y[i] != 1) and (s_conf[i] >= t):
                            s_trades.append(s_y[i] == y_true[i])
                            last_s = i
                    # Ensemble
                    if i >= last_e + cooldown:
                        if (g_y[i] == s_y[i]) and (g_y[i] != 1) and (g_conf[i] >= t) and (s_conf[i] >= t):
                            e_trades.append(g_y[i] == y_true[i])
                            last_e = i
                
                g_acc = np.mean(g_trades) if g_trades else 0
                s_acc = np.mean(s_trades) if s_trades else 0
                e_acc = np.mean(e_trades) if e_trades else 0
                
                print(f"{symbol:<10} | {t:<6.0%} | {g_acc:<12.1%} | {s_acc:<12.1%} | {e_acc:<12.1%} | {len(e_trades)}")
            print("-" * 140)
            
            # Cleanup
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Audit failed for {symbol}: {e}")

    print("="*140)

if __name__ == "__main__":
    run_ultimate_audit()
