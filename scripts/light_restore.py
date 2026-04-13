
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), 'signals.db')
WEEK_START = "2026-02-09 00:00:00"

def restore_signals_light():
    print(f"Connecting to DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. Count signals to be restored
        query = """
        SELECT count(*) FROM signals 
        WHERE timestamp >= ? 
        AND signal IN ('BUY', 'SELL')
        """
        try:
            count = cursor.execute(query, (WEEK_START,)).fetchone()[0]
        except Exception as e:
            print(f"Query Error: {e}")
            return

        if count == 0:
            print("No signals found for this week to restore.")
            conn.close()
            return

        print(f"Found {count} BUY/SELL signals since {WEEK_START}.")
        print("Forcing status to 'ACTIVE'...")
        
        # 2. Update to ACTIVE
        update_q = """
        UPDATE signals 
        SET outcome = 'ACTIVE' 
        WHERE timestamp >= ? 
        AND signal IN ('BUY', 'SELL')
        """
        cursor.execute(update_q, (WEEK_START,))
        conn.commit()
        
        modified = cursor.rowcount
        print(f"Successfully resurrected {modified} signals to ACTIVE.")
        
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    restore_signals_light()
