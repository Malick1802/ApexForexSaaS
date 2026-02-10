
import sys
import os
import pandas as pd
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline import DataEngine
from core.inference import InferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_freshness(symbol):
    print(f"\n--- Checking freshness for {symbol} ---")
    data_engine = DataEngine()
    
    # Fetch data
    df = data_engine.fetch(symbol, interval="1h", days=5)
    
    if df is not None and not df.empty:
        last_candle = df.index[-1]
        now_utc = pd.Timestamp.now(tz='UTC')
        
        # Localize if naive
        if last_candle.tzinfo is None:
            last_candle_aware = last_candle.tz_localize('UTC')
        else:
            last_candle_aware = last_candle.tz_convert('UTC')
            
        diff = now_utc - last_candle_aware
        hours_diff = diff.total_seconds() / 3600.0
        
        print(f"Last Candle: {last_candle}")
        print(f"Last Candle (Aware): {last_candle_aware}")
        print(f"Now UTC: {now_utc}")
        print(f"Hours Diff: {hours_diff:.4f} hours")
        
        inf_engine = InferenceEngine()
        is_stale = inf_engine._is_data_stale(last_candle)
        print(f"InferenceEngine._is_data_stale says: {is_stale}")
        
    else:
        print("Failed to fetch data or empty df")

if __name__ == "__main__":
    check_freshness("EURUSD")
    check_freshness("EURJPY")
