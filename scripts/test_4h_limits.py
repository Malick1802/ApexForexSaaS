
import yfinance as yf

def test_4h_limits(symbol="AUDNZD=X"):
    print(f"Testing yfinance limits for {symbol} (Interval: 4h)...")
    
    # Test 5 Years
    print("\n--- Attempting 5 Years (Max/5y) ---")
    try:
        # yfinance often allows longer history for larger intervals.
        # But '4h' might be considered 'intraday' (capped at 730d) or 'daily' (uncapped).
        # Let's find out.
        df = yf.download(symbol, period="5y", interval="1h", progress=False) # Wait, intent was 4h? 
        # But yfinance standard intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # It does NOT natively support '4h'.
        # We would have to start with '1h' and resample.
        # So we are stuck with the 1h limit (730d).
        pass
    except: pass
    
    print("Wait... yfinance doesn't support '4h' natively. Checking '1d'...")
    try:
        df = yf.download(symbol, period="10y", interval="1d", progress=False)
        if not df.empty:
            print(f"Got {len(df)} rows (1d).")
            print(f"Range: {df.index.min()} to {df.index.max()}")
        else:
            print("Empty DataFrame.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_4h_limits()
