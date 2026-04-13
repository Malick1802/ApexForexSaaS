
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def check_active_stacking():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT symbol, count(*) as count 
    FROM signals 
    WHERE outcome = 'ACTIVE' 
    AND signal IN ('BUY', 'SELL')
    GROUP BY symbol 
    HAVING count > 1
    ORDER BY count DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No stacking detected. Each pair has at most 1 active signal.")
    else:
        print(f"DETECTED STACKING! Found {len(df)} pairs with multiple active signals:")
        print(df)
        print(f"\nTotal excess signals: {df['count'].sum() - len(df)}")

if __name__ == "__main__":
    check_active_stacking()
