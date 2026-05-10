import sqlite3
import os

dbs = ['signals.db', 'dashboard/signals.db', 'data/signals.db']
for db in dbs:
    if os.path.exists(db):
        try:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), outcome FROM signals WHERE symbol='CADCHF' GROUP BY outcome")
            results = cursor.fetchall()
            print(f"DB: {db}")
            if not results:
                print("  No CADCHF signals found.")
            for count, outcome in results:
                print(f"  Outcome: {outcome}, Count: {count}")
            conn.close()
        except Exception as e:
            print(f"Error checking {db}: {e}")
