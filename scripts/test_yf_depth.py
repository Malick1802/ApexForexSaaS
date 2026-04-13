
import yfinance as yf
from datetime import datetime, timedelta

def test_yf_depth():
    print("--- Testing yfinance 5-Year Depth ---")
    symbol = "EURUSD=X"
    # yfinance limit for 1h is usually 730 days. Let's check.
    try:
        data = yf.download(symbol, period="5y", interval="1h")
        if data.empty:
            print("❌ EURUSD=X (1h) Failed for 5 years")
        else:
            print(f"✅ EURUSD=X (1h) Success: {len(data)} rows")
            print(f"Earliest: {data.index[0]}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Check 4h
    try:
        data = yf.download(symbol, period="5y", interval="1h") # yf doesn't have 4h easily, uses 60m then resamples
        print(f"Checking 1d for comparison...")
        data_d = yf.download(symbol, period="5y", interval="1d")
        print(f"✅ EURUSD=X (1d) Success: {len(data_d)} rows")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_yf_depth()
