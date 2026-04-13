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
        
        # Query for AUDNZD BUY/SELL signals in the last 7 days
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        query = "SELECT * FROM signals WHERE symbol = 'AUDNZD' AND signal IN ('BUY', 'SELL') ORDER BY timestamp DESC"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"--- All BUY/SELL signals for AUDNZD ---")
        if not rows:
            print("No BUY/SELL signals found at all.")
        else:
            df = pd.DataFrame(rows, columns=columns)
            print(df.to_string())
        
        conn.close()
    except Exception as e:
        print(f"Error querying signals: {e}")

if __name__ == '__main__':
    query_signals()
