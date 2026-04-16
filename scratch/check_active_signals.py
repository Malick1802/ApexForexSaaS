import sqlite3
import pandas as pd

def check_active_signals():
    conn = sqlite3.connect('signals.db')
    query_active = "SELECT id, symbol, signal, outcome, is_hidden, timestamp FROM signals WHERE outcome = 'ACTIVE'"
    df_active = pd.read_sql_query(query_active, conn)
    print("--- CURRENTLY ACTIVE SIGNALS ---")
    if df_active.empty:
        print("No active signals! Ghost trades resolved.")
    else:
        print(df_active.to_string(index=False))
        
    print("\n--- RECENTLY RESOLVED SIGNALS (LAST 10) ---")
    query_resolved = "SELECT id, symbol, signal, outcome, timestamp FROM signals WHERE outcome IN ('SUCCESS', 'FAIL') ORDER BY timestamp DESC LIMIT 10"
    df_resolved = pd.read_sql_query(query_resolved, conn)
    if df_resolved.empty:
        print("No resolved signals.")
    else:
        print(df_resolved.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    check_active_signals()
