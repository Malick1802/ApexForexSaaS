import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def query_signals():
    try:
        conn = sqlite3.connect('signals.db')
        # Get column names
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(signals)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Query for AUDNZD signals in the last 24 hours
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        query = "SELECT * FROM signals WHERE symbol = 'AUDNZD' AND timestamp >= ? ORDER BY timestamp DESC"
        cursor.execute(query, (yesterday,))
        rows = cursor.fetchall()
        
        print(f"--- Recent signals for AUDNZD (since {yesterday}) ---")
        if not rows:
            print("No signals found.")
        else:
            df = pd.DataFrame(rows, columns=columns)
            print(df.to_string())
        
        conn.close()
    except Exception as e:
        print(f"Error querying signals: {e}")

if __name__ == '__main__':
    query_signals()
