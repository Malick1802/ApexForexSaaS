
import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), 'signals.db')
CUTOFF_DATE = "2026-02-09 00:00:00"

def purge_old_signals():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Count signals to be deleted
    query = "SELECT count(*) FROM signals WHERE timestamp < ?"
    count = cursor.execute(query, (CUTOFF_DATE,)).fetchone()[0]
    
    if count == 0:
        print(f"No signals found older than {CUTOFF_DATE}.")
        conn.close()
        return

    print(f"Found {count} signals older than {CUTOFF_DATE}.")
    print("Deleting...")
    
    # 2. Delete
    delete_q = "DELETE FROM signals WHERE timestamp < ?"
    cursor.execute(delete_q, (CUTOFF_DATE,))
    conn.commit()
    
    deleted = cursor.rowcount
    print(f"Successfully deleted {deleted} signals.")
    
    # 3. Verify remaining
    remaining = cursor.execute("SELECT count(*) FROM signals").fetchone()[0]
    print(f"Remaining signals in DB: {remaining}")
    
    conn.close()

if __name__ == "__main__":
    purge_old_signals()
