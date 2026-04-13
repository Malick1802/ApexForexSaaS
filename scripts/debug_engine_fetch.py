
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from data_pipeline.engine import DataEngine

def check_engine_fetch():
    print("--- Debugging DataEngine Fetch ---")
    try:
        engine = DataEngine()
        
        # Mimic app.py exact call
        print("Fetching USDCHF (1h) with use_cache=False...")
        df = engine.fetch("USDCHF", interval="1h", days=60, use_cache=False)
        
        if df.empty:
            print("❌ Dataframe is EMPTY")
            return

        print(f"Index TZ: {df.index.tz}")
        print("Tail:")
        print(df.tail())
        
        last_ts = df.index[-1]
        print(f"Last Candle: {last_ts}")
        
        # Check diff logic
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
        else:
            last_ts = last_ts.tz_convert('UTC')
            
        now = pd.Timestamp.now(tz='UTC')
        diff = (now - last_ts).total_seconds() / 3600
        print(f"Diff Hours from Now: {diff:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_engine_fetch()
