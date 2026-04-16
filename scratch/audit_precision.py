import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone

def audit_precision_updates():
    conn = sqlite3.connect('signals.db')
    
    # 1. Check for signals resolved in the last 24 hours
    print("--- SIGNALS RESOLVED IN LAST 24H ---")
    query_resolved = """
    SELECT id, symbol, signal, outcome, confidence, timestamp, is_hidden 
    FROM signals 
    WHERE outcome IN ('SUCCESS', 'FAIL') 
    AND timestamp > datetime('now', '-24 hours')
    ORDER BY timestamp DESC
    """
    df_resolved = pd.read_sql_query(query_resolved, conn)
    if df_resolved.empty:
        print("No signals resolved in the last 24 hours.")
    else:
        print(df_resolved.to_string(index=False))

    # 2. Check for ACTIVE signals (Waiting to be resolved)
    print("\n--- CURRENTLY ACTIVE SIGNALS (WATCHDOG PENDING) ---")
    query_active = """
    SELECT id, symbol, signal, outcome, is_hidden, timestamp 
    FROM signals 
    WHERE outcome = 'ACTIVE' 
    ORDER BY timestamp DESC 
    LIMIT 10
    """
    df_active = pd.read_sql_query(query_active, conn)
    print(df_active.to_string(index=False))

    # 3. Check the Performance Matrix counts
    print("\n--- PERFORMANCE MATRIX (TOP 10 BY RELEVANCE) ---")
    try:
        query_matrix = "SELECT * FROM performance_metrics ORDER BY total_trades DESC LIMIT 10"
        df_matrix = pd.read_sql_query(query_matrix, conn)
        print(df_matrix.to_string(index=False))
    except Exception as e:
        print(f"Error reading performance_metrics: {e}")
        # Try checking for common tables if that one doesn't exist
        print("Checking all tables in DB...")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(cursor.fetchall())

    conn.close()

if __name__ == "__main__":
    audit_precision_updates()
