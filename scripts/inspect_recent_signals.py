
import sqlite3
import pandas as pd
import os

DB_PATH = "signals.db"

def inspect_recent():
    if not os.path.exists(DB_PATH):
        print(f"DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT id, timestamp, symbol, signal, outcome, confidence, model_trades FROM signals ORDER BY id DESC LIMIT 20"
        df = pd.read_sql_query(query, conn)
        print(df.to_string())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_recent()
