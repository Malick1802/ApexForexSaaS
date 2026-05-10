import sqlite3
import json

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('SELECT id, symbol, signal, confidence, confidence_tier, is_hidden, outcome, timestamp FROM signals ORDER BY timestamp DESC LIMIT 20')
rows = [dict(r) for r in cur.fetchall()]
print(json.dumps(rows, indent=2))
conn.close()
