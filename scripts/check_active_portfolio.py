
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

def check_portfolio():
    print("--- 🕵️ Checking Active Portfolio ---")
    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()
    
    # Get all active signals
    cursor.execute("SELECT id, symbol, signal, price_at_signal, tp_price, sl_price, timestamp FROM signals WHERE outcome='ACTIVE'")
    active_rows = cursor.fetchall()
    
    if not active_rows:
        print("No active signals found.")
        conn.close()
        return

    print(f"Found {len(active_rows)} active signals. verifying with market data...")
    
    updates = 0
    
    for row in active_rows:
        sig_id, symbol, signal, entry, tp, sl, ts = row
        print(f"\nChecking {symbol} ({signal}) ID: {sig_id}...")
        
        # Yahoo Finance Ticker
        # Handle conversion if needed (e.g. standard pairs usually work)
        yf_symbol = f"{symbol}=X"
        
        try:
            # Fetch last 2 days to cover the hold period
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="2d", interval="5m")
            
            if df.empty:
                print(f"⚠️ No data for {symbol}")
                continue
                
            # Filter data AFTER the signal timestamp
            sig_ts = pd.to_datetime(ts)
            # Localize signal ts to match yfinance (usually UTC or provider local)
            # Simple check: filter by index > sig_ts
            if df.index.tz:
                sig_ts = sig_ts.tz_localize(df.index.tz)
            
            # Slice data from signal entry onwards
            relevant_data = df[df.index >= sig_ts]
            
            if relevant_data.empty:
               print("Data too old or mismatch.")
               continue
               
            highs = relevant_data['High'].max()
            lows = relevant_data['Low'].min()
            
            outcome = None
            
            if signal == 'BUY':
                if lows <= sl:
                    outcome = 'FAIL'
                    print(f"❌ SL HIT! Low {lows:.5f} <= {sl:.5f}")
                elif highs >= tp:
                    outcome = 'SUCCESS'
                    print(f"🎯 TP HIT! High {highs:.5f} >= {tp:.5f}")
            elif signal == 'SELL':
                if highs >= sl:
                    outcome = 'FAIL'
                    print(f"❌ SL HIT! High {highs:.5f} >= {sl:.5f}")
                elif lows <= tp:
                    outcome = 'SUCCESS'
                    print(f"🎯 TP HIT! Low {lows:.5f} <= {tp:.5f}")
            
            if outcome:
                print(f"🔄 Updating DB to {outcome}...")
                cursor.execute("UPDATE signals SET outcome=? WHERE id=?", (outcome, sig_id))
                updates += 1
            else:
                 print(f"✅ Still Active. (Range: {lows:.4f} - {highs:.4f})")
                 
        except Exception as e:
            print(f"Error checking {symbol}: {e}")
            
    conn.commit()
    conn.close()
    
    print(f"\n--- Update Complete. {updates} records fixed. ---")

if __name__ == "__main__":
    check_portfolio()
