
import sqlite3
import pandas as pd
import os

DB_PATH = "signals.db"

def inspect_trades():
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT id, time(timestamp), symbol, signal, model_trades FROM signals WHERE symbol='AUDCAD' AND signal='BUY' ORDER BY id DESC LIMIT 1"
        df = pd.read_sql_query(query, conn)
        print("Data from DB:")
        print(df.to_string())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_trades()
