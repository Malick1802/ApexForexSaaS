
import sqlite3
import pandas as pd
import os

DB_PATH = "signals.db"

def check_duplicates():
    if not os.path.exists(DB_PATH):
        print(f"DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        # Check for duplicates in the last 24 hours
        # We group by symbol, signal, and a generic 5-minute time window to catch "bursts"
        query = """
        SELECT 
            symbol, 
            signal, 
            strftime('%Y-%m-%d %H:%M', timestamp) as minute,
            COUNT(*) as count 
        FROM signals 
        WHERE timestamp > datetime('now', '-24 hours') 
          AND signal IN ('BUY', 'SELL')
        GROUP BY symbol, signal, minute
        HAVING count > 1
        ORDER BY count DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No DB duplicates found in the last 24h.")
        else:
            print("❌ DUPLICATES FOUND IN DB:")
            print(df.to_string())
            
        print("-" * 40)
        print("Last 10 Signals:")
        q2 = "SELECT id, symbol, signal, timestamp FROM signals WHERE signal IN ('BUY', 'SELL') ORDER BY id DESC LIMIT 10"
        df2 = pd.read_sql_query(q2, conn)
        print(df2.to_string())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_duplicates()
