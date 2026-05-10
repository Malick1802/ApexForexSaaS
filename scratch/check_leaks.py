import sqlite3
import os
from datetime import datetime

db_path = "signals.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()
# Check for any BUY/SELL signals after 12:16 UTC with confidence < 0.6
c.execute("SELECT id, symbol, signal, confidence, status, timestamp FROM signals WHERE timestamp > '2026-05-04T12:16:00' AND confidence < 0.6 AND signal IN ('BUY', 'SELL')")
rows = c.fetchall()
if rows:
    print(f"FOUND {len(rows)} LEAKED SIGNALS:")
    for r in rows:
        print(r)
else:
    print("No leaked signals found after 12:16 UTC.")
conn.close()
