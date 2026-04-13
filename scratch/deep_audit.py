import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Fix pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import InferenceEngine
from data_pipeline.engine import DataEngine
from core.mt5_connector import get_mt5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEEP-AUDIT")

def audit():
    logger.info("=== APEX DEEP AUDIT STARTING ===")
    
    # 1. MT5 Status
    logger.info("--- 1. MT5 Connection ---")
    mt5 = get_mt5()
    if not mt5:
        logger.error("MT5 Connector failed to initialize.")
    else:
        info = mt5.account_info()
        if info:
            logger.info(f"Account: {info.login}, Balance: {info.balance}, Trade Allowed: {info.trade_allowed}")
        else:
            logger.error("Could not fetch account info. MT5 Terminal might be closed.")

    # 2. Data Integrity
    logger.info("--- 2. Data Fetching ---")
    engine = DataEngine()
    symbol = "EURUSD"
    df = engine.fetch(symbol, interval="1h", days=5)
    if df is not None and not df.empty:
        logger.info(f"Fetched {len(df)} bars for {symbol}. Last Close: {df['close'].iloc[-1]}, Last Time: {df.index[-1]}")
    else:
        logger.error(f"Failed to fetch data for {symbol}")
        return

    # 3. Inference & Feature Alignment
    logger.info("--- 3. Inference Engine Audit ---")
    inf = InferenceEngine()
    
    # Manually run parts of predict_symbol to see internal states
    try:
        models = inf.load_phase3_expert(symbol) or inf.load_foundation_model(symbol)
        if not models:
            logger.error("No Foundation/Expert model found.")
        else:
            logger.info(f"Model Type: {models.get('model_type')}")
            scaler = models.get('scaler')
            logger.info(f"Scaler Features expected: {scaler.n_features_in_}")
            
            # Predict
            res = inf.predict_symbol(symbol, save_to_db=False)
            if res:
                logger.info(f"Prediction: {res.get('signal')} at {res.get('confidence', 0):.2%} confidence")
                logger.info(f"Conviction Bias Check: Raw Buy={res.get('buy_prob'):.4f}, Raw Sell={res.get('sell_prob'):.4f}")
                if res.get('is_locked'):
                    logger.warning("Symbol is currently LOCKED in Database.")
            else:
                logger.warning("Inference returned None. Check staleness/regime/data.")
    except Exception as e:
        logger.exception(f"Inference Audit Failed: {e}")

    logger.info("=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    audit()
