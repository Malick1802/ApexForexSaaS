import sqlite3
import pandas as pd

def audit_active_details():
    conn = sqlite3.connect('signals.db')
    query = "SELECT id, symbol, signal, price_at_signal, tp_price, sl_price, timestamp FROM signals WHERE outcome = 'ACTIVE'"
    df = pd.read_sql_query(query, conn)
    print("--- ACTIVE SIGNAL DETAILS ---")
    print(df.to_string(index=False))
    conn.close()

if __name__ == "__main__":
    audit_active_details()
