import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Project Root Resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "signals.db")

def analyze_conviction_stats():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Total Signals Distribution
    query = "SELECT confidence FROM signals WHERE signal IN ('BUY', 'SELL')"
    df = pd.read_sql_query(query, conn)
    
    print("--- Conviction Distribution (All History) ---")
    if len(df) > 0:
        # Scale to percentages if they are floats
        if df['confidence'].max() <= 1.0:
            df['confidence'] = df['confidence'] * 100
            
        bins = [0, 50, 60, 70, 80, 90, 101]
        labels = ['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        df['tier'] = pd.cut(df['confidence'], bins=bins, labels=labels)
        
        counts = df['tier'].value_counts().sort_index()
        total = len(df)
        for tier, count in counts.items():
            print(f"{tier}: {count} signals ({count/total:.1%})")
    else:
        print("No signals found in database.")

    # 2. Daily Frequency (Last 7 Days)
    query_daily = """
        SELECT date(timestamp) as day, COUNT(*) as count 
        FROM signals 
        WHERE signal IN ('BUY', 'SELL') 
        AND timestamp >= date('now', '-7 days')
        GROUP BY day
    """
    df_daily = pd.read_sql_query(query_daily, conn)
    
    print("\n--- Daily Signal Volume (Last 7 Days) ---")
    if len(df_daily) > 0:
        print(df_daily.to_string(index=False))
        print(f"\nAverage Signals Per Day: {df_daily['count'].mean():.1f}")
    else:
        print("No recent signals found.")

    # 3. High Conviction Frequency (60%+)
    query_high = """
        SELECT date(timestamp) as day, COUNT(*) as count 
        FROM signals 
        WHERE signal IN ('BUY', 'SELL') 
        AND confidence >= 0.60
        AND timestamp >= date('now', '-7 days')
        GROUP BY day
    """
    df_high = pd.read_sql_query(query_high, conn)
    
    print("\n--- High Conviction (60%+) Volume (Last 7 Days) ---")
    if len(df_high) > 0:
        print(df_high.to_string(index=False))
        print(f"\nAverage 60%+ Signals Per Day: {df_high['count'].mean():.1f}")
    else:
        print("No 60%+ signals found recently.")

    conn.close()

if __name__ == "__main__":
    analyze_conviction_stats()
