
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def check_tickets():
    conn = sqlite3.connect(DB_PATH)
    # Check schema first to see if mt5_ticket exists
    try:
        # Check null tickets
        query = "SELECT count(*) FROM signals WHERE outcome='ACTIVE' AND (mt5_ticket IS NULL OR mt5_ticket='')"
        count = conn.execute(query).fetchone()[0]
        print(f"Active Signals with NULL Ticket (Pending Execution): {count}")
        
        # Check Total Active
        total = conn.execute("SELECT count(*) FROM signals WHERE outcome='ACTIVE'").fetchone()[0]
        print(f"Total Active Signals: {total}")
        
    except Exception as e:
        print(f"Error checking tickets: {e}")
        # Maybe column doesn't exist?
        
    conn.close()

if __name__ == "__main__":
    check_tickets()
