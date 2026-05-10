import sqlite3
import os

db_path = "signals.db"
if not os.path.exists(db_path):
    print("DB not found")
    exit()

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT id, symbol, signal, confidence, status, timestamp, is_hidden, expert_signal FROM signals WHERE symbol='EURNZD' ORDER BY timestamp DESC LIMIT 3")
rows = c.fetchall()
for r in rows:
    print(r)
conn.close()
