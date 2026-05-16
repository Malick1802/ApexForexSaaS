import os, sys, logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mt5_connector import get_mt5
from models.foundation_trainer_v3 import fetch_mt5_pair, fetch_yf_macro, build_base_features, add_global_context_v3, rolling_zscore
from models.global_brain import GatedResidualNetwork, VariableSelectionNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V3_Verify")

def verify_brain():
    # 1. Load Model
    model_path = PROJECT_ROOT / "models" / "foundation_v3" / "foundation_brain.keras"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    logger.info("Loading v3 Global Brain...")
    custom_objs = {
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork
    }
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objs)
    
    # 2. Fetch Live Data
    mt5 = get_mt5()
    if not mt5: return
    
    pairs = ["EURUSD", "GOLD", "GBPUSD"]
    macro_tickers = {
        "SP500": "^GSPC", "OIL": "CL=F", "NASDAQ": "^IXIC",
        "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX",
        "DXY": "DX-Y.NYB", "COPPER": "HG=F", "BTC": "BTC-USD"
    }
    
    logger.info("Fetching live macro context (48h)...")
    raw = {}
    for p in pairs:
        raw[p] = fetch_mt5_pair(mt5, p, 10) 
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, 15)
        if not df.empty: raw[k] = df

    # 3. Align
    common = None
    for df in raw.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
    
    # 4. Predict
    logger.info("="*50)
    logger.info("V3 GLOBAL BRAIN LIVE PREDICTIONS")
    logger.info("="*50)
    
    for symbol in pairs:
        try:
            feat = build_base_features(aligned[symbol])
            feat = add_global_context_v3(feat, aligned, feat.index)
            
            # Use last 48 hours
            for col in feat.columns:
                feat[col] = rolling_zscore(feat[col])
            
            input_data = feat.values[-48:].reshape(1, SEQ_LEN if 'SEQ_LEN' in globals() else 48, -1).astype(np.float32)
            
            preds = model.predict(input_data, verbose=0)[0]
            classes = ["SELL", "WAIT", "BUY"]
            best_idx = np.argmax(preds)
            
            logger.info(f"{symbol:8} | Action: {classes[best_idx]:5} | Conf: {preds[best_idx]:.2%} [S:{preds[0]:.1%} W:{preds[1]:.1%} B:{preds[2]:.1%}]")
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")

if __name__ == "__main__":
    verify_brain()
