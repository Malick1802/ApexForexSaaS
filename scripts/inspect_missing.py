
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def inspect_missing_symbols():
    conn = sqlite3.connect(DB_PATH)
    symbols = ['EURUSD', 'AUDCAD', 'USDSGD', 'EURGBP']
    
    for sym in symbols:
        print(f"--- {sym} ---")
        query = "SELECT * FROM signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT 5"
        df = pd.read_sql_query(query, conn, params=(sym,))
        if df.empty:
            print("  No records found.")
        else:
            print(df[['id', 'signal', 'timestamp', 'outcome', 'price_at_signal']])
    
    conn.close()

if __name__ == "__main__":
    inspect_missing_symbols()
