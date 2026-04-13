
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def revert_closed_trades():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Identify the 7 signals (Success/Fail) from this week
    now = datetime.now()
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    
    query = """
    SELECT id, symbol, signal, outcome, timestamp 
    FROM signals 
    WHERE outcome IN ('SUCCESS', 'FAIL') 
    AND timestamp >= ?
    """
    
    df = pd.read_sql_query(query, conn, params=(start_of_week.isoformat(),))
    
    if df.empty:
        print("No closed trades found to revert.")
        conn.close()
        return

    print("Found the following 'Closed' trades that are likely Open in MT5:")
    print(df)
    
    # 2. Revert them to ACTIVE
    ids_to_revert = df['id'].tolist()
    
    # Update query
    update_q = f"""
    UPDATE signals 
    SET outcome = 'ACTIVE' 
    WHERE id IN ({','.join(['?']*len(ids_to_revert))})
    """
    
    cursor.execute(update_q, ids_to_revert)
    conn.commit()
    
    print(f"\nSuccessfully reverted {cursor.rowcount} signals to 'ACTIVE'.")
    print("These should now appear in the 'Active Signals' list on the dashboard.")
    
    conn.close()

if __name__ == "__main__":
    revert_closed_trades()
