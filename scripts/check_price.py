
import yfinance as yf
import pandas as pd

def check_nzdjpy():
    print("Fetching NZDJPY data...")
    try:
        # Fetch data
        ticker = yf.Ticker("NZDJPY=X")
        df = ticker.history(period="1d", interval="5m")
        
        if df.empty:
            print("No data fetched.")
            return

        print(f"Data Points: {len(df)}")
        print(f"Last Price: {df['Close'].iloc[-1]:.5f}")
        
        # SL Level from DB: 93.117 (SELL)
        sl_level = 93.117
        tp_level = 91.877
        
        max_high = df['High'].max()
        min_low = df['Low'].min()
        
        print(f"Max High (last 24h): {max_high:.5f}")
        print(f"Min Low (last 24h): {min_low:.5f}")
        
        if max_high >= sl_level:
            print(f"❌ SL HIT! High {max_high:.5f} >= {sl_level}")
        elif min_low <= tp_level:
             print(f"🎯 TP HIT! Low {min_low:.5f} <= {tp_level}")
        else:
            print("Trade still within limits.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_nzdjpy()
