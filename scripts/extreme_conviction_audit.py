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
logger = logging.getLogger("ExtremeAudit")

def run_extreme_audit(pairs=["GOLD", "GBPUSD", "USDJPY", "AUDUSD", "EURUSD"], days=30):
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

    thresholds = [0.52, 0.70, 0.80, 0.90, 0.95]
    
    print("\n" + "="*160)
    print(f" EXTREME CONVICTION AUDIT (Last {days} Days)")
    print("="*160)
    header = f"{'PAIR':<10} | {'MODEL':<10} | " + " | ".join([f"{t:<15.0%}" for t in thresholds])
    print(header)
    print("-" * 160)
    
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
            
            # 5. Evaluate Global
            g_preds = global_model.predict(X, verbose=0)
            g_y_pred = np.argmax(g_preds, axis=1)
            g_conf = np.max(g_preds, axis=1)
            
            row_g = f"{symbol:<10} | GLOBAL     | "
            for t in thresholds:
                mask = (g_conf >= t) & (g_y_pred != 1)
                if np.any(mask):
                    acc = np.mean(g_y_pred[mask] == y_true[mask])
                    row_g += f"{acc:<7.1%} ({np.sum(mask):<3}) | "
                else:
                    row_g += f"{'N/A':<15} | "
            print(row_g)
            
            # 6. Evaluate Specialist
            s_preds = spec_model.predict(X, verbose=0)
            s_y_pred = np.argmax(s_preds, axis=1)
            s_conf = np.max(s_preds, axis=1)
            
            row_s = f"{'':<10} | SPECIALIST | "
            for t in thresholds:
                mask = (s_conf >= t) & (s_y_pred != 1)
                if np.any(mask):
                    acc = np.mean(s_y_pred[mask] == y_true[mask])
                    row_s += f"{acc:<7.1%} ({np.sum(mask):<3}) | "
                else:
                    row_s += f"{'N/A':<15} | "
            print(row_s)
            print("-" * 160)
            
            # Clean up
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Audit failed for {symbol}: {e}")

    print("="*160)

if __name__ == "__main__":
    run_extreme_audit()
