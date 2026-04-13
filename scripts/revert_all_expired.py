
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def revert_all_expired():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Count Expired Signals
    query = "SELECT count(*) FROM signals WHERE outcome = 'EXPIRED' AND signal IN ('BUY', 'SELL')"
    count = cursor.execute(query).fetchone()[0]
    
    if count == 0:
        print("No expired signals found to revert.")
        conn.close()
        return

    print(f"Found {count} EXPIRED signals (BUY/SELL only).")
    print("Reverting them to ACTIVE status...")
    
    # 2. Update to ACTIVE
    update_q = "UPDATE signals SET outcome = 'ACTIVE' WHERE outcome = 'EXPIRED' AND signal IN ('BUY', 'SELL')"
    cursor.execute(update_q)
    conn.commit()
    
    print(f"Successfully resurrected {cursor.rowcount} signals.")
    print("These are now ACTIVE and will be monitored until they hit TP/SL.")
    
    conn.close()

if __name__ == "__main__":
    revert_all_expired()
