
import yfinance as yf
import pandas as pd
import sqlite3
import json
from datetime import datetime, timezone

def check_tz():
    print("--- YFinance Timezone Check ---")
    df = yf.Ticker("USDCHF=X").history(period="1d", interval="1h")
    print(f"Index TZ: {df.index.tz}")
    print("Tail:")
    print(df.tail())
    
    last = df.index[-1]
    now = pd.Timestamp.now(tz='UTC')
    
    # If naive, assume UTC and check diff
    if last.tzinfo is None:
        last_utc_assumption = last.tz_localize('UTC')
        diff = (now - last_utc_assumption).total_seconds() / 3600
        print(f"\nIf strictly UTC naive: Diff is {diff:.2f} hours")
        
        # If naive was actually NY (UTC-5)
        last_ny_assumption = last.tz_localize('America/New_York')
        diff_ny = (now - last_ny_assumption).total_seconds() / 3600
        print(f"If actually NY (UTC-5): Diff is {diff_ny:.2f} hours")

def check_db():
    print("\n--- DB Active Signal Check ---")
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM signals WHERE outcome='ACTIVE' AND symbol='USDCHF'")
    row = c.fetchone()
    
    if row:
        d = dict(row)
        print(f"buy_prob (Col): {d.get('buy_prob')}")
        print(f"raw_prob (Str): {d.get('raw_probabilities')}")
        
        # Simulation of database.py parsing logic
        raw = d.get('raw_probabilities')
        if raw and raw != '[]':
            try:
                probs = json.loads(raw)
                print(f"Parsed raw: {probs}")
                if len(probs) == 3:
                    print(f"Legacy logic would set buy_prob to: {probs[1]}")
            except:
                print("Parse failed")
    else:
        print("No active USDCHF signal")
    conn.close()

if __name__ == "__main__":
    check_tz()
    check_db()
