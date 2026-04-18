import sqlite3
import os

db_path = 'signals.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Find all AUDUSD signals from the last 24h
    cursor.execute("SELECT id, symbol, signal, confidence_tier, outcome, status, timestamp, is_hidden FROM signals WHERE symbol = 'AUDUSD' AND timestamp > '2026-04-17' ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    print("--- AUDUSD SIGNAL TIMELINE ---")
    for r in rows:
        print(f"ID: {r['id']} | Signal: {r['signal']} | Tier: {r['confidence_tier']}% | Outcome: {r['outcome']} | Status: {r['status']} | Time: {r['timestamp']}")
    conn.close()
else:
    print("DB missing.")
