import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import SignalDatabase
import json

db = SignalDatabase()
with db._get_connection() as conn:
    conn.row_factory = lambda cursor, row: dict(zip([col[0] for col in cursor.description], row))
    cursor = conn.cursor()
    
    # 1. Check GOLD signals
    gold = cursor.execute('SELECT * FROM signals WHERE symbol IN ("GOLD", "XAUUSD") ORDER BY timestamp DESC LIMIT 5').fetchall()
    print("--- LATEST GOLD SIGNALS ---")
    for s in gold:
        print(f"ID: {s['id']} | {s['symbol']} | {s['signal']} | {s['outcome']} | {s['timestamp']}")

    # 2. Check ACTIVE signals
    active = cursor.execute('SELECT * FROM signals WHERE outcome="ACTIVE"').fetchall()
    print("\n--- ALL ACTIVE SIGNALS ---")
    for s in active:
        print(f"ID: {s['id']} | {s['symbol']} | {s['signal']} | {s['timestamp']}")
