import sqlite3
import os

db_path = r'c:\Users\artem\Downloads\ApexForexSaaS\data\signals.db'
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT count(*) FROM signals")
        count = cursor.fetchone()[0]
        print(f"Signals count: {count}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
