import sqlite3
import pandas as pd
import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import SignalDatabase

db = SignalDatabase()
conn = sqlite3.connect(db.db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== USDSGD DUPLICATE BUG HUNT ===")
cursor.execute("SELECT id, signal, status, outcome, is_hidden, confidence_tier, timestamp FROM signals WHERE symbol = 'USDSGD' ORDER BY timestamp DESC LIMIT 5")
rows = cursor.fetchall()

for r in rows:
    tier = r['confidence_tier']
    print(f"ID: {r['id']} | {r['signal']} | Outcome: {r['outcome']} | Status: {r['status']} | Tier: {tier} | Time: {r['timestamp']}")

conn.close()
