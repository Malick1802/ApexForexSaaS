
import yfinance as yf
from datetime import datetime, timedelta

def test_limits(symbol="AUDNZD=X"):
    print(f"Testing yfinance limits for {symbol} (Interval: 1h)...")
    
    # Test 2 Years
    print("\n--- Attempting 730 Days (2 Years) ---")
    try:
        df = yf.download(symbol, period="730d", interval="1h", progress=False)
        if not df.empty:
            print(f"Got {len(df)} rows.")
            print(f"Range: {df.index.min()} to {df.index.max()}")
        else:
            print("Empty DataFrame.")
    except Exception as e:
        print(f"Error: {e}")

    # Test 5 Years
    print("\n--- Attempting 5 Years (Max/5y) ---")
    try:
        # yfinance often requires 'max' for extended history, or specific dates
        # but 1h is usually capped. Let's try 'max' first.
        df = yf.download(symbol, period="5y", interval="1h", progress=False)
        if not df.empty:
            print(f"Got {len(df)} rows.")
            print(f"Range: {df.index.min()} to {df.index.max()}")
        else:
            print("Empty DataFrame.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_limits()
