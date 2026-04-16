import sqlite3
import pandas as pd
from datetime import datetime, timezone

def check_nzdjpy():
    conn = sqlite3.connect('signals.db')
    # Get everything from today
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    query = f"""
    SELECT id, symbol, signal, confidence, raw_confidence, is_hidden, timestamp 
    FROM signals 
    WHERE symbol = 'NZDJPY' 
    AND timestamp LIKE '{today}%'
    ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    print(f"--- NZDJPY SIGNALS FOR {today} ---")
    if df.empty:
        print("No signals found in database for today.")
    else:
        print(df.to_string(index=False))
    conn.close()

if __name__ == "__main__":
    check_nzdjpy()
