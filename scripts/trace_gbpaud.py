
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

def trace_gbpaud():
    print("--- Tracing GBPAUD Signal ---")
    conn = sqlite3.connect("signals.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM signals WHERE symbol='GBPAUD' ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    
    if not row:
        print("GBPAUD signal not found.")
        conn.close()
        return

    sig = dict(row)
    print(f"ID: {sig['id']}")
    print(f"Outcome: {sig['outcome']}")
    print(f"Timestamp (Raw): {sig['timestamp']}")
    
    # Simulate Logic
    try:
        ts = pd.to_datetime(sig['timestamp'])
        now = datetime.now()
        cutoff = now - timedelta(hours=48)
        
        print(f"Parsed TS: {ts}")
        print(f"Current Time: {now}")
        print(f"Cutoff (48h ago): {cutoff}")
        
        # Check comparison
        is_recent = ts >= cutoff
        print(f"Is Recent (>= Cutoff)? {is_recent}")
        
        # Check Outcome
        is_completed = sig['outcome'] in ['SUCCESS', 'FAIL']
        print(f"Is Completed? {is_completed}")
        
        # Check Win
        is_win = sig['outcome'] == 'SUCCESS'
        print(f"Is Win? {is_win}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    trace_gbpaud()
