
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8') 

def analyze_period():
    # Define time range: From yesterday start (2026-02-11)
    start_date = "2026-02-11 00:00:00"
    print(f"--- Performance Analysis (Since {start_date}) ---")
    
    try:
        conn = sqlite3.connect("signals.db")
        cursor = conn.cursor()
        
        # Query ALL trades in range
        query = """
        SELECT symbol, signal, outcome, timestamp, confidence 
        FROM signals 
        WHERE timestamp >= ?
        AND signal IN ('BUY', 'SELL') -- Filter out WAIT signals to see actionable trades
        ORDER BY timestamp DESC
        """
        cursor.execute(query, (start_date,))
        rows = cursor.fetchall()
        
        if not rows:
            print("No completed trades found in this period.")
            conn.close()
            return

        cols = ['Symbol', 'Type', 'Outcome', 'Time', 'Conf']
        df = pd.DataFrame(rows, columns=cols)
        completed = df[df['Outcome'].isin(['SUCCESS', 'FAIL'])]
        active = df[df['Outcome'] == 'ACTIVE']
        
        total_trades = len(df)
        total_completed = len(completed)
        total_active = len(active)
        
        wins = len(completed[completed['Outcome'] == 'SUCCESS'])
        losses = len(completed[completed['Outcome'] == 'FAIL'])
        
        # Win Rate based on COMPLETED trades only
        win_rate = (wins / total_completed * 100) if total_completed > 0 else 0.0
        
        print(f"\nTotal Actionable Signals: {total_trades}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        print(f"⏳ Active: {total_active}")
        print(f"📉 Realized Win Rate: {win_rate:.1f}% (of {total_completed} closed)")
        
        print("\n--- Trade History (Yesterday to Today) ---")
        print(df[['Symbol', 'Type', 'Outcome', 'Time']].to_string(index=False))
        
        print("\n--- Breakdown by Pair ---")
        pair_stats = df.groupby('Symbol')['Outcome'].value_counts().unstack().fillna(0)
        print(pair_stats)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_period()
