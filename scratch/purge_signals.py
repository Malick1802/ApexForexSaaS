import sqlite3
import os

db_path = 'signals.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE signals SET outcome='EXPIRED' WHERE outcome='ACTIVE'")
    conn.commit()
    print(f"Purged {cur.rowcount} stale active signals.")
    conn.close()
else:
    print("❌ signals.db not found.")
