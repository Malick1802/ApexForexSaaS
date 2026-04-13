import sqlite3
import os

def check_wal():
    db_path = "signals.db"
    if not os.path.exists(db_path):
        print(f"❌ DB not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode")
    mode = cursor.fetchone()[0]
    conn.close()

    print(f"--- 📊 SQLite Performance Check ---")
    if mode.lower() == "wal":
        print(f"✅ SUCCESS: Journal Mode is {mode.upper()}")
        print("   (Optimized for Azure/Linux concurrency)")
    else:
        print(f"❌ FAILURE: Journal Mode is {mode.upper()}")

if __name__ == "__main__":
    check_wal()
