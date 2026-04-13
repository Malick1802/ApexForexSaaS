
import sqlite3
import pandas as pd
import yfinance as yf
import os

DB_PATH = "signals.db"
SYMBOL = "AUDCAD=X"

def compare_signals():
    conn = sqlite3.connect(DB_PATH)
    
    # Get last 2 non-WAIT signals for AUDCAD
    query = """
    SELECT id, timestamp, signal, price_at_signal, outcome 
    FROM signals 
    WHERE symbol IN ('AUDCAD', 'AUDCAD=X') AND signal IN ('BUY', 'SELL')
    ORDER BY id DESC LIMIT 5
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("Recent AUDCAD Signals:")
    print(df.to_string())
    
    # Get current price
    print(f"\nFetching current price for {SYMBOL}...")
    try:
        ticker = yf.Ticker(SYMBOL)
        # Fast fetch
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            data = ticker.history(period="5d", interval="1h")
            
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            print(f"Current Price: {current_price:.5f}")
            
            # Analyze
            for _, row in df.iterrows():
                sig_type = row['signal']
                entry = row['price_at_signal']
                
                if sig_type == "BUY":
                    pnl = (current_price - entry) * 10000
                else:
                    pnl = (entry - current_price) * 10000
                    
                print(f"Signal {row['id']} ({sig_type} @ {entry:.5f}): PnL ≈ {pnl:.1f} pips")
        else:
            print("Could not fetch current price.")
            
    except Exception as e:
        print(f"Error fetching price: {e}")

if __name__ == "__main__":
    compare_signals()
