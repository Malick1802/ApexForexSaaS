
import sys
import os
import pandas as pd
import sqlite3
import yfinance as yf
import json
from datetime import datetime, timezone

def check_market_data(symbol="USDCHF"):
    print(f"--- Checking Market Data for {symbol} (Direct yfinance) ---")
    try:
        # Fetch 1h data
        ticker = yf.Ticker(f"{symbol}=X")
        df = ticker.history(period="5d", interval="1h")
        
        if df.empty:
            print("❌ Dataframe is EMPTY")
            return

        last_ts = df.index[-1]
        print(f"Last Candle Timestamp (Raw): {last_ts}")
        
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
        else:
            last_ts = last_ts.tz_convert('UTC')
            
        print(f"Last Candle Timestamp (UTC): {last_ts}")
        
        now_utc = pd.Timestamp.now(tz='UTC')
        print(f"Current Time (UTC): {now_utc}")
        
        diff_hours = (now_utc - last_ts).total_seconds() / 3600.0
        print(f"Diff Hours: {diff_hours:.4f}")
        
        if diff_hours > 4.0:
            print("⛔ Result: MARKET CLOSED (True)")
        else:
            print("✅ Result: MARKET OPEN (False)")
            
    except Exception as e:
        print(f"Error checking market data: {e}")

def check_db_signal(symbol="USDCHF"):
    print(f"\n--- Checking DB Signal for {symbol} (Direct sqlite3) ---")
    try:
        conn = sqlite3.connect('signals.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Get latest active signal
        c.execute("SELECT * FROM signals WHERE symbol=? AND outcome='ACTIVE' ORDER BY timestamp DESC LIMIT 1", (symbol,))
        row = c.fetchone()
        
        if not row:
            print("No active signal found.")
            return

        print("Active Signal Keys:", row.keys())
        print(f"Signal: {row['signal']}")
        
        # Check for prob keys
        keys = ['buy_prob', 'sell_prob', 'wait_prob']
        for k in keys:
            if k in row.keys():
                print(f"{k}: {row[k]}")
            else:
                print(f"❌ '{k}' NOT FOUND in DB columns")
                
        conn.close()
    except Exception as e:
        print(f"Error checking DB: {e}")

if __name__ == "__main__":
    check_market_data()
    check_db_signal()
