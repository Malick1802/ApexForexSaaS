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
logger = logging.getLogger("SoloAudit")

def run_solo_audit(symbol="EURUSD", days=14):
    mt5 = get_mt5()
    
    # 1. Fetch Context
    macro_tickers = {"SP500": "^GSPC", "OIL": "CL=F", "DXY": "DX-Y.NYB"}
    raw = {symbol: fetch_mt5_pair(mt5, symbol, days + 5)}
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, days + 10)
        if not df.empty: raw[k] = df

    common = None
    for df in raw.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
    
    # 2. Features
    feat = build_base_features(aligned[symbol])
    feat = add_global_context_v3(feat, aligned, feat.index)
    for col in feat.columns: feat[col] = rolling_zscore(feat[col])
    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
    labels = triple_barrier_label(aligned[symbol].reindex(feat.index))
    
    # 3. Sequences
    X, y_true = [], []
    feat_vals = feat.values.astype(np.float32)
    label_vals = labels.values.astype(np.int32)
    for i in range(48, len(feat_vals) - 24):
        X.append(feat_vals[i-48:i])
        y_true.append(label_vals[i])
    X = np.array(X); y_true = np.array(y_true)
    
    # 4. Load Specialist
    spec_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
    model = tf.keras.models.load_model(str(spec_path), custom_objects={'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork})
    
    # 5. Predict
    preds = model.predict(X, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)
    
    print("\n" + "="*60)
    print(f" {symbol} SPECIALIST SOLO PERFORMANCE (Last {days} Days)")
    print("="*60)
    print(f"{'THRESHOLD':<15} | {'WIN RATE':<15} | {'TRADES'}")
    print("-" * 60)
    
    for t in [0.52, 0.60, 0.65, 0.70]:
        mask = (confs >= t) & (y_pred != 1)
        if np.any(mask):
            acc = np.mean(y_pred[mask] == y_true[mask])
            print(f"{t:<15.0%} | {acc:<15.1%} | {np.sum(mask)}")
        else:
            print(f"{t:<15.0%} | {'N/A':<15} | 0")
    print("="*60)

if __name__ == "__main__":
    run_solo_audit()
