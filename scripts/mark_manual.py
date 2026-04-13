
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def mark_current_signals_manual():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Count pending signals (ACTIVE + NULL ticket)
    query_check = "SELECT count(*) FROM signals WHERE outcome='ACTIVE' AND (mt5_ticket IS NULL OR mt5_ticket='')"
    pending_count = cursor.execute(query_check).fetchone()[0]
    
    if pending_count == 0:
        print("No pending signals found to mark as MANUAL.")
        conn.close()
        return

    print(f"Found {pending_count} pending signals to skip.")
    print("Marking as 'MANUAL'...")
    
    # 2. Update
    query_update = """
    UPDATE signals 
    SET mt5_ticket = 'MANUAL' 
    WHERE outcome = 'ACTIVE' 
    AND (mt5_ticket IS NULL OR mt5_ticket='')
    """
    cursor.execute(query_update)
    conn.commit()
    
    updated = cursor.rowcount
    print(f"Successfully marked {updated} signals as MANUAL.")
    
    conn.close()

if __name__ == "__main__":
    mark_current_signals_manual()
