
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def check_earliest_signal():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT min(timestamp) FROM signals"
    min_ts = pd.read_sql_query(query, conn).iloc[0,0]
    print(f"Earliest Signal Timestamp: {min_ts}")
    conn.close()

if __name__ == "__main__":
    check_earliest_signal()
