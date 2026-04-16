import sqlite3
import pandas as pd

def check_crude():
    conn = sqlite3.connect('signals.db')
    query = """
    SELECT id, symbol, signal, expert_signal, confidence, regime, is_hidden, is_proven, timestamp 
    FROM signals 
    WHERE symbol = 'CrudeOIL' 
    ORDER BY timestamp DESC 
    LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No CrudeOIL signals found.")
        return
        
    print("--- CRUDE OIL SIGNAL HISTORY ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    check_crude()
