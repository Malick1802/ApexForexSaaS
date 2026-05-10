import sqlite3
import os

db_path = 'signals.db'
if not os.path.exists(db_path):
    print("Database not found.")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Strictly keep only OOS_AUDIT baseline and BUY/SELL signals
cursor.execute("""
    DELETE FROM signals 
    WHERE expert_signal != 'OOS_AUDIT' 
    AND signal NOT IN ('BUY', 'SELL')
""")

print(f"Final Purge: Removed {cursor.rowcount} non-trading records.")
conn.commit()
conn.close()
