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
logger = logging.getLogger("HeadToHead")

def run_comparison(pairs=["GOLD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"], days=14):
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

    print("\n" + "="*110)
    print(f" HEAD-TO-HEAD AUDIT: GLOBAL BRAIN vs SPECIALIST (Last {days} Days)")
    print("="*110)
    print(f"{'PAIR':<10} | {'GLOBAL ACC':<15} | {'SPEC ACC':<15} | {'SPEC SNIPER (60%)':<20} | {'VOL (G/S/SNIP)'}")
    print("-" * 110)
    
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
            
            # 5. Global Predictions
            g_preds = global_model.predict(X, verbose=0)
            g_y_pred = np.argmax(g_preds, axis=1)
            g_mask = (g_y_pred != 1)
            g_acc = np.mean(g_y_pred[g_mask] == y_true[g_mask]) if np.any(g_mask) else 0
            g_vol = np.sum(g_mask)
            
            # 6. Specialist Predictions
            s_preds = spec_model.predict(X, verbose=0)
            s_y_pred = np.argmax(s_preds, axis=1)
            
            # Spec Base (52%)
            s_mask = (s_y_pred != 1)
            s_acc = np.mean(s_y_pred[s_mask] == y_true[s_mask]) if np.any(s_mask) else 0
            s_vol = np.sum(s_mask)
            
            # Spec Sniper (60%)
            snip_mask = (np.max(s_preds, axis=1) >= 0.60) & (s_y_pred != 1)
            snip_acc = np.mean(s_y_pred[snip_mask] == y_true[snip_mask]) if np.any(snip_mask) else 0
            snip_vol = np.sum(snip_mask)
            
            print(f"{symbol:<10} | {g_acc:<15.1%} | {s_acc:<15.1%} | {snip_acc:<20.1%} | {g_vol}/{s_vol}/{snip_vol}")
            
            # Clean up
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Comparison failed for {symbol}: {e}")

    print("="*110)

if __name__ == "__main__":
    run_comparison()
