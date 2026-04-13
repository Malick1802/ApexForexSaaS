
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from data_pipeline.engine import DataEngine

def test_twelvedata_depth():
    print("--- Testing TwelveData 5-Year Depth (Free Tier) ---")
    try:
        engine = DataEngine(provider_name="twelvedata")
        
        # Test 1: Try 5 years (1825 days)
        print("Fetching EURUSD (1h) - Goal: 5 Years (1825 days)...")
        df = engine.fetch("EURUSD", interval="1h", days=1825, use_cache=False)
        
        if df.empty:
            print("❌ Dataframe is EMPTY")
        else:
            print(f"✅ SUCCESS: Fetched {len(df)} rows")
            print(f"Earliest: {df.index[0]}")
            print(f"Latest:   {df.index[-1]}")
            days_fetched = (df.index[-1] - df.index[0]).days
            print(f"Total History Span: {days_fetched} days")
            
            if days_fetched < 1800:
                 print(f"⚠️ TwelveData Limited: Requested 1825 days but only got {days_fetched} days.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_twelvedata_depth()
