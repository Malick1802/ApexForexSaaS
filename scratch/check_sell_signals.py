import sqlite3
import pandas as pd

def check_sell_imbalance():
    conn = sqlite3.connect('signals.db')
    
    # 1. Look for ANY sell signals in the last 7 days
    print("--- SELL SIGNAL COUNT (LAST 7 DAYS) ---")
    query = """
    SELECT symbol, outcome, COUNT(*) as count 
    FROM signals 
    WHERE signal = 'SELL' 
    AND timestamp > datetime('now', '-7 days')
    GROUP BY symbol, outcome
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        print("CRITICAL: Zero SELL signals generated in the last 7 days across all pairs.")
    else:
        print(df.to_string(index=False))

    # 2. Compare to BUY signals
    print("\n--- BUY SIGNAL COUNT (LAST 7 DAYS) ---")
    query_buy = """
    SELECT symbol, outcome, COUNT(*) as count 
    FROM signals 
    WHERE signal = 'BUY' 
    AND timestamp > datetime('now', '-7 days')
    GROUP BY symbol, outcome
    """
    df_buy = pd.read_sql_query(query_buy, conn)
    print(df_buy.to_string(index=False))

    # 3. Check for currently ACTIVE sgnals
    print("\n--- CURRENTLY ACTIVE SIGNALS ---")
    query_active = "SELECT id, symbol, signal, is_hidden, timestamp FROM signals WHERE outcome = 'ACTIVE'"
    df_active = pd.read_sql_query(query_active, conn)
    if df_active.empty:
        print("No active signals.")
    else:
        print(df_active.to_string(index=False))
        
    conn.close()

if __name__ == "__main__":
    check_sell_imbalance()
