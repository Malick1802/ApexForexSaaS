
import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), 'signals.db')
WEEK_START = "2026-02-09 00:00:00"

def restore_signals():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Count signals to be restored
    # We target ALL BUY/SELL signals from this week, regardless of current status (completed or expired)
    # The Sentinel will sort them out in the next pass.
    query = """
    SELECT count(*) FROM signals 
    WHERE timestamp >= ? 
    AND signal IN ('BUY', 'SELL')
    """
    count = cursor.execute(query, (WEEK_START,)).fetchone()[0]
    
    if count == 0:
        print("No signals found for this week to restore.")
        conn.close()
        return

    print(f"Found {count} BUY/SELL signals since {WEEK_START}.")
    print("Forcing status to 'ACTIVE' for re-evaluation...")
    
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
    print("Please ensure 'sentinel.py' is running to process outcomes based on price.")

if __name__ == "__main__":
    restore_signals()
