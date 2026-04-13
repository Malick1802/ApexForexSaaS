
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def count_active():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Count Active BUY/SELL
        query = "SELECT count(*) FROM signals WHERE outcome='ACTIVE' AND signal IN ('BUY', 'SELL')"
        count = cursor.execute(query).fetchone()[0]
        
        print(f"Active Signals: {count}")
        
        # Verify timestamps range
        ts_q = "SELECT min(timestamp), max(timestamp) FROM signals WHERE outcome='ACTIVE'"
        min_ts, max_ts = cursor.execute(ts_q).fetchone()
        print(f"Time Range: {min_ts} to {max_ts}")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    count_active()
