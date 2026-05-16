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
logger = logging.getLogger("EnsembleSniper")

def run_ensemble_sniper_audit(days=14):
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

    pairs = ["GOLD", "GBPUSD", "EURUSD", "AUDUSD", "USDJPY"]
    thresholds = [0.52, 0.60, 0.70]
    
    print("\n" + "="*100)
    print(f" ENSEMBLE SNIPER AUDIT (Agreement + High Conviction) (Last {days} Days)")
    print("="*100)
    print(f"{'PAIR':<10} | {'52% AGREE':<15} | {'60% AGREE':<15} | {'70% AGREE':<15}")
    print("-" * 100)
    
    for symbol in pairs:
        try:
            # 1. Load Specialist
            spec_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
            if not spec_path.exists(): continue
            spec_model = tf.keras.models.load_model(str(spec_path), custom_objects=custom_objs)
            
            # 2. Fetch & Process
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
            
            # 3. Predict Both
            g_preds = global_model.predict(X, verbose=0)
            g_y = np.argmax(g_preds, axis=1)
            g_conf = np.max(g_preds, axis=1)
            
            s_preds = spec_model.predict(X, verbose=0)
            s_y = np.argmax(s_preds, axis=1)
            s_conf = np.max(s_preds, axis=1)
            
            # 4. Analyze thresholds
            row = f"{symbol:<10} | "
            for t in thresholds:
                # AGREEMENT + BOTH meet threshold
                mask = (g_y == s_y) & (g_y != 1) & (g_conf >= t) & (s_conf >= t)
                if np.any(mask):
                    acc = np.mean(g_y[mask] == y_true[mask])
                    row += f"{acc:<7.1%} ({np.sum(mask):<3}) | "
                else:
                    row += f"{'N/A':<15} | "
            print(row)
            
            # Cleanup
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Audit failed for {symbol}: {e}")

    print("="*100)

if __name__ == "__main__":
    run_ensemble_sniper_audit()
