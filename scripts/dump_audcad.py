
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def dump_audcad():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM signals WHERE symbol = 'AUDCAD' ORDER BY timestamp DESC LIMIT 20"
    df = pd.read_sql_query(query, conn)
    print("Recent AUDCAD Signals:")
    print(df[['id', 'timestamp', 'signal', 'outcome']])
    conn.close()

if __name__ == "__main__":
    dump_audcad()
