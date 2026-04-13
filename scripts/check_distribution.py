
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def check_status_distribution():
    print(f"Checking DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT outcome, signal, count(*) as count FROM signals GROUP BY outcome, signal"
    
    df = pd.read_sql_query(query, conn)
    print(df)
    
    conn.close()

if __name__ == "__main__":
    check_status_distribution()
