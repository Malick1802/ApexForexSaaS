
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def debug_missing_signals():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Check EXPIRED count
    exp_count = pd.read_sql_query("SELECT count(*) FROM signals WHERE outcome = 'EXPIRED'", conn).iloc[0,0]
    print(f"Current EXPIRED count: {exp_count}")
    
    # 2. Check for the AUDCAD trade (approx time 2026-02-12 18:45)
    # User screenshot says 18:45:01. Let's look for AUDCAD on that day.
    query = """
    SELECT * FROM signals 
    WHERE symbol = 'AUDCAD' 
    AND timestamp LIKE '2026-02-12%'
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("Specific AUDCAD trade not found in DB!")
    else:
        print("\nFound AUDCAD signals on 2026-02-12:")
        print(df[['id', 'symbol', 'signal', 'timestamp', 'outcome']])

    conn.close()

if __name__ == "__main__":
    debug_missing_signals()
