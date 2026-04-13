
import sys
import os
from pathlib import Path
import logging
import pandas as pd

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_pipeline import DataEngine
from core.inference import InferenceEngine

def debug_fetch():
    print(f"--- DIAGNOSTIC FETCH FOR EURUSD ---")
    
    # 1. Initialize Engines
    data_engine = DataEngine()
    inference = InferenceEngine()
    
    symbol = "EURUSD"
    
    # 2. Test Fetch
    print(f"\n[1] Fetching {symbol} (1h, 5 days)...")
    df = data_engine.fetch(symbol, interval="1h", days=5)
    
    if df is None or df.empty:
        print("[FAIL] FETCH FAILED: Returned None or Empty DataFrame")
        return
        
    print(f"[OK] Data Fetched: {len(df)} rows")
    last_candle = df.index[-1]
    last_close = df['close'].iloc[-1]
    print(f"   Last Candle: {last_candle}")
    print(f"   Last Close: {last_close}")
    
    # 3. Test Stale Check
    print(f"\n[2] Testing Staleness Logic...")
    is_stale = inference._is_data_stale(last_candle)
    print(f"   Is Stale? {'YES (BLOCKING)' if is_stale else 'NO (FRESH)'}")
    
    # ... (skipping some lines) ...
    
    # 5. Model Load Check
    print(f"\n[3] Model Availability Check (90% Expert)...")
    models = inference.load_models(symbol, win_rate=90)
    if models:
        print(f"[OK] Models Loaded. Trades Volume: {models.get('trades', 'N/A')}")
    else:
        print("[FAIL] Models NOT Found for 90%")

if __name__ == "__main__":
    try:
        debug_fetch()
    except Exception as e:
        print(f"\n[CRITICAL] CRASH: {e}")
        import traceback
        traceback.print_exc()
