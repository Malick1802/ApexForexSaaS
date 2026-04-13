
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def purge_closed_signals():
    print(f"Connecting to DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Count before purge
        q_count = "SELECT count(*) FROM signals WHERE outcome != 'ACTIVE'"
        count_before = cursor.execute(q_count).fetchone()[0]
        
        if count_before == 0:
            print("No closed signals found to purge.")
            conn.close()
            return

        print(f"Found {count_before} signals to purge (SUCCESS, FAIL, EXPIRED, WAIT, etc)")
        
        # 2. Delete non-ACTIVE
        q_delete = "DELETE FROM signals WHERE outcome != 'ACTIVE'"
        cursor.execute(q_delete)
        deleted = cursor.rowcount
        conn.commit()
        
        print(f"Successfully deleted {deleted} closed signals.")
        
        # Verify remaining
        q_active = "SELECT count(*) FROM signals"
        remaining = cursor.execute(q_active).fetchone()[0]
        print(f"Remaining Signals (ACTIVE only): {remaining}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error purging DB: {e}")

if __name__ == "__main__":
    purge_closed_signals()
