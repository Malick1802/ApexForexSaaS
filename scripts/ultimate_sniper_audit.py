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
logger = logging.getLogger("UltimateAudit")

def run_ultimate_audit(pairs=["GOLD", "GBPUSD", "USDJPY", "AUDUSD", "EURUSD"], days=14):
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
    
    # Load Global Brain
    global_model = tf.keras.models.load_model(str(PROJECT_ROOT / "models" / "foundation_v3" / "foundation_brain.keras"), custom_objects=custom_objs)

    print("\n" + "="*125)
    print(f" ULTIMATE SNIPER AUDIT: GLOBAL vs SPECIALIST (Last {days} Days)")
    print("="*125)
    print(f"{'PAIR':<10} | {'GLOBAL 52%':<12} | {'GLOBAL 60%':<12} | {'SPEC 52%':<12} | {'SPEC 60%':<12} | {'VOL (G52/G60/S52/S60)'}")
    print("-" * 125)
    
    for symbol in pairs:
        try:
            # 1. Load Specialist
            spec_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
            if not spec_path.exists(): continue
            spec_model = tf.keras.models.load_model(str(spec_path), custom_objects=custom_objs)
            
            # 2. Fetch & Align
            raw = {symbol: fetch_mt5_pair(mt5, symbol, days + 5)}
            raw.update(macro_raw)
            common = None
            for df in raw.values():
                common = df.index if common is None else common.intersection(df.index)
            aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
            
            # 3. Features
            feat = build_base_features(aligned[symbol])
            feat = add_global_context_v3(feat, aligned, feat.index)
            for col in feat.columns: feat[col] = rolling_zscore(feat[col])
            feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
            labels = triple_barrier_label(aligned[symbol].reindex(feat.index))
            
            # 4. Sequences
            X, y_true = [], []
            feat_vals = feat.values.astype(np.float32)
            label_vals = labels.values.astype(np.int32)
            for i in range(48, len(feat_vals) - 24):
                X.append(feat_vals[i-48:i])
                y_true.append(label_vals[i])
            X = np.array(X); y_true = np.array(y_true)
            
            # 5. Global Results
            g_preds = global_model.predict(X, verbose=0)
            g_y_pred = np.argmax(g_preds, axis=1)
            
            g52_mask = (g_y_pred != 1)
            g52_acc = np.mean(g_y_pred[g52_mask] == y_true[g52_mask]) if np.any(g52_mask) else 0
            g60_mask = (np.max(g_preds, axis=1) >= 0.60) & (g_y_pred != 1)
            g60_acc = np.mean(g_y_pred[g60_mask] == y_true[g60_mask]) if np.any(g60_mask) else 0
            
            # 6. Specialist Results
            s_preds = spec_model.predict(X, verbose=0)
            s_y_pred = np.argmax(s_preds, axis=1)
            
            s52_mask = (s_y_pred != 1)
            s52_acc = np.mean(s_y_pred[s52_mask] == y_true[s52_mask]) if np.any(s52_mask) else 0
            s60_mask = (np.max(s_preds, axis=1) >= 0.60) & (s_y_pred != 1)
            s60_acc = np.mean(s_y_pred[s60_mask] == y_true[s60_mask]) if np.any(s60_mask) else 0
            
            # Volumes
            vols = f"{np.sum(g52_mask)}/{np.sum(g60_mask)}/{np.sum(s52_mask)}/{np.sum(s60_mask)}"
            
            print(f"{symbol:<10} | {g52_acc:<12.1%} | {g60_acc:<12.1%} | {s52_acc:<12.1%} | {s60_acc:<12.1%} | {vols}")
            
            # Clean up
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Ultimate Audit failed for {symbol}: {e}")

    print("="*125)

if __name__ == "__main__":
    run_ultimate_audit()
