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
logger = logging.getLogger("FinalAudit")

def run_final_audit(days=14, cooldown=24):
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
    all_pairs = [p.name for p in spec_base.iterdir() if (p / "specialist_brain.keras").exists()]
    
    results = []
    
    print(f"\nScanning {len(all_pairs)} pairs... This will take a moment.")
    
    for symbol in all_pairs:
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
            
            # 4. Predictions
            g_preds = global_model.predict(X, verbose=0)
            s_preds = spec_model.predict(X, verbose=0)
            g_y, g_conf = np.argmax(g_preds, axis=1), np.max(g_preds, axis=1)
            s_y, s_conf = np.argmax(s_preds, axis=1), np.max(s_preds, axis=1)
            
            # 5. Sparse Simulation (Agreement @ 60%)
            last_idx = -cooldown
            trades = []
            for i in range(len(X)):
                if i >= last_idx + cooldown:
                    if (g_y[i] == s_y[i]) and (g_y[i] != 1) and (g_conf[i] >= 0.60) and (s_conf[i] >= 0.60):
                        trades.append(g_y[i] == y_true[i])
                        last_idx = i
            
            acc = np.mean(trades) if trades else 0
            vol = len(trades)
            
            # Also get Solo Specialist @ 60% for comparison
            last_idx_s = -cooldown
            trades_s = []
            for i in range(len(X)):
                if i >= last_idx_s + cooldown:
                    if (s_y[i] != 1) and (s_conf[i] >= 0.60):
                        trades_s.append(s_y[i] == y_true[i])
                        last_idx_s = i
            acc_s = np.mean(trades_s) if trades_s else 0
            vol_s = len(trades_s)

            results.append({
                "PAIR": symbol,
                "ENS_ACC": f"{acc:.1%}" if trades else "N/A",
                "ENS_VOL": vol,
                "SPEC_ACC": f"{acc_s:.1%}" if trades_s else "N/A",
                "SPEC_VOL": vol_s
            })
            
            # Cleanup
            del spec_model, X, y_true, feat, labels, aligned, raw
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Audit failed for {symbol}: {e}")

    # 6. Final Table Output
    df_final = pd.DataFrame(results)
    print("\n" + "="*100)
    print(f" FINAL GLOBAL FLEET AUDIT: REAL-WORLD SPARSITY (Last 14 Days)")
    print("="*100)
    print(df_final.to_string(index=False))
    print("="*100)

if __name__ == "__main__":
    run_final_audit()
