
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def analyze_db():
    print(f"Checking DB at: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("DB not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Calculate Start of Week (Monday)
    now = datetime.now()
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"Start of Week (Monday): {start_of_week}")
    
    query = """
    SELECT symbol, signal, outcome, timestamp 
    FROM signals 
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(start_of_week.isoformat(),))
    conn.close()
    
    if df.empty:
        print("No signals found for this week.")
        return

    print(f"Total Signals since Monday: {len(df)}")
    
    print("\n--- Outcome Distribution ---")
    print(df['outcome'].value_counts())
    
    print("\n--- Signal Type Distribution ---")
    print(df['signal'].value_counts())
    
    print("\n--- Outcome by Signal Type ---")
    print(df.groupby(['signal', 'outcome']).size())

    # Simulate Dashboard Counters
    # Active
    active = df[df['outcome'] == 'ACTIVE']
    print(f"\nDashboard 'Active' Estimate: {len(active)}")
    
    # Win Rate Closed (Success/Fail)
    completed = df[df['outcome'].isin(['SUCCESS', 'FAIL'])]
    print(f"Dashboard 'Win Rate Closed' Estimate: {len(completed)}")
    
    # Expired/Closed List (Non-Active BUY/SELL)
    # Exclude WAIT
    expired_list = df[
        (df['outcome'] != 'ACTIVE') & 
        (df['signal'].isin(['BUY', 'SELL']))
    ]
    print(f"Dashboard 'Expired/Closed' Estimate: {len(expired_list)}")
    
    # Check for anomalies
    print("\n--- Sample of 'EXPIRED' signals ---")
    expired_only = df[df['outcome'] == 'EXPIRED'].head(5)
    print(expired_only)

if __name__ == "__main__":
    analyze_db()
