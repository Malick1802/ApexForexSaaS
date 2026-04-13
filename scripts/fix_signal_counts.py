
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')
WEEK_START = "2026-02-09 00:00:00"

def fix_signals():
    print(f"Connecting to DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. DELETE ACTIVE WAIT signals
        print("Checking for ACTIVE WAIT signals...")
        cursor.execute("DELETE FROM signals WHERE outcome = 'ACTIVE' AND signal = 'WAIT'")
        deleted_wait = cursor.rowcount
        print(f"Deleted {deleted_wait} invalid 'ACTIVE WAIT' signals.")

        # 2. RESURRECT EXPIRED BUY/SELL signals
        print("Checking for EXPIRED BUY/SELL signals...")
        update_q = """
        UPDATE signals 
        SET outcome = 'ACTIVE' 
        WHERE timestamp >= ? 
        AND signal IN ('BUY', 'SELL') 
        AND outcome = 'EXPIRED'
        """
        cursor.execute(update_q, (WEEK_START,))
        resurrected = cursor.rowcount
        print(f"Resurrected {resurrected} EXPIRED signals to ACTIVE.")

        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_signals()
