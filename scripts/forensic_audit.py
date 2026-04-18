import sqlite3
import os
from datetime import datetime, timedelta

db_path = 'signals.db'
if not os.path.exists(db_path):
    print("Database not found.")
else:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check all signals from the last 3 days
    # We look for anything that is NOT specifically 'SUCCESS', 'FAIL', 'EXPIRED'
    query = """
    SELECT id, symbol, signal, confidence_tier, outcome, status, timestamp, is_hidden
    FROM signals 
    WHERE timestamp > ?
    ORDER BY timestamp DESC
    """
    three_days_ago = (datetime.now() - timedelta(days=3)).isoformat()
    cursor.execute(query, (three_days_ago,))
    rows = cursor.fetchall()
    
    print(f"--- FORENSIC AUDIT (Last 72 Hours) ---")
    active_candidates = []
    
    for r in rows:
        # What constitutes a "Missing" trade?
        # 1. outcome is 'ACTIVE' or 'N/A' or 'PENDING'
        # 2. signal is BUY or SELL
        if r['outcome'] in ['ACTIVE', 'N/A', 'PENDING', 'SENT'] and r['signal'] in ['BUY', 'SELL']:
            active_candidates.append(r)
        
        # Also look for anything that was 'SENT' to Telegram but maybe has a weird outcome
        if r['status'] == 'SENT' and r not in active_candidates:
            active_candidates.append(r)

    if active_candidates:
        print(f"Found {len(active_candidates)} potential active/sent signals:")
        print(f"| ID | Symbol | Sig | Tier | Outcome | Status | Timestamp | Hidden |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for c in active_candidates:
            print(f"| {c['id']} | {c['symbol']} | {c['signal']} | {c['confidence_tier']}% | {c['outcome']} | {c['status']} | {c['timestamp'][:16]} | {c['is_hidden']} |")
    else:
        print("No candidates found in the last 72 hours.")
    
    conn.close()
