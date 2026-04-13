import sqlite3
import os

db_path = 'signals.db'
if not os.path.exists(db_path):
    print("Database not found.")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Search for all GOLD signals in the history
cursor.execute("SELECT signal, outcome, timestamp FROM signals WHERE symbol = 'GOLD' ORDER BY timestamp DESC")
rows = cursor.fetchall()
conn.close()

if not rows:
    print("No GOLD signals found in this database.")
else:
    print(f"Total GOLD history found: {len(rows)} entries.")
    for row in rows[:20]:
        print(f"{row['timestamp']} | {row['signal']} | {row['outcome']}")
