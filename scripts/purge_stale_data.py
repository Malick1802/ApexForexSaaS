import sqlite3
import os
import sys
from datetime import datetime, timedelta

# Project Root
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
DB_PATH = os.path.join(_ROOT, "signals.db")

def purge():
    print(f"Opening database: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("ERROR: Database file not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Get count of crisis signals
    cursor.execute("SELECT COUNT(*) FROM signals WHERE regime = 'CRISIS'")
    count = cursor.fetchone()[0]
    print(f"Found {count} CRISIS signals in database.")

    # 2. Delete signals older than 24 hours OR all crisis signals to force fresh scan
    # We choose to delete ALL crisis signals to ensure no ghost data remains.
    print("Purging all CRISIS signals to force fresh recalibrated scan...")
    cursor.execute("DELETE FROM signals WHERE regime = 'CRISIS'")
    deleted = cursor.rowcount
    
    # 3. Also delete very old signals (older than 3 days)
    three_days_ago = (datetime.now() - timedelta(days=3)).isoformat()
    cursor.execute("DELETE FROM signals WHERE timestamp < ?", (three_days_ago,))
    
    conn.commit()
    conn.close()
    print(f"SUCCESS: Deleted {deleted} CRISIS records. Dashboard will now wait for fresh data.")

if __name__ == "__main__":
    purge()
