import os, sys, logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mt5_connector import get_mt5
from models.foundation_trainer_v3 import (
    fetch_mt5_pair, fetch_yf_macro, build_base_features, 
    add_global_context_v3, rolling_zscore
)
from models.global_brain import GatedResidualNetwork, VariableSelectionNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveTestV3")

def run_live_test(symbols=["GOLD", "GBPUSD"]):
    mt5 = get_mt5()
    if not mt5: return
    
    # 1. Fetch Real-time Context (Last 72 hours for safety)
    logger.info("Fetching live market context...")
    macro_tickers = {
        "SP500": "^GSPC", "OIL": "CL=F", "NASDAQ": "^IXIC",
        "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX",
        "DXY": "DX-Y.NYB", "COPPER": "HG=F", "BTC": "BTC-USD"
    }
    raw = {}
    for k, v in macro_tickers.items():
        df = fetch_yf_macro(k, v, 10)
        if not df.empty: raw[k] = df
        
    custom_objs = {'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork}

    for symbol in symbols:
        logger.info(f"\n--- TESTING {symbol} SPECIALIST ---")
        try:
            # 2. Load Model
            spec_path = PROJECT_ROOT / "models" / "specialist" / symbol / "specialist_brain.keras"
            model = tf.keras.models.load_model(str(spec_path), custom_objects=custom_objs)
            
            # 3. Process Pair Data
            pair_df = fetch_mt5_pair(mt5, symbol, 10)
            raw[symbol] = pair_df
            
            common = None
            for df in raw.values():
                common = df.index if common is None else common.intersection(df.index)
            aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
            
            feat = build_base_features(aligned[symbol])
            feat = add_global_context_v3(feat, aligned, feat.index)
            for col in feat.columns: feat[col] = rolling_zscore(feat[col])
            feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
            
            # 4. Final Sequence (Last 48 bars)
            X = feat.values[-48:].astype(np.float32)
            X = np.expand_dims(X, axis=0) # Batch size 1
            
            # 5. Predict
            preds = model.predict(X, verbose=0)[0]
            classes = ["SELL", "WAIT", "BUY"]
            top_class = np.argmax(preds)
            confidence = preds[top_class]
            
            logger.info(f"LIVE SIGNAL: {classes[top_class]}")
            logger.info(f"CONFIDENCE:  {confidence:.2%}")
            
            if confidence > 0.60 and classes[top_class] != "WAIT":
                logger.info("💎 [SNIPER ALERT] High-Confidence Entry Detected!")
            else:
                logger.info("⏳ [NO TRADE] Sentiment currently neutral or uncertain.")
                
        except Exception as e:
            logger.error(f"Test failed for {symbol}: {e}")

if __name__ == "__main__":
    run_live_test()
