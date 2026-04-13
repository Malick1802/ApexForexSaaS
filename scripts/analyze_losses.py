
import sqlite3
import pandas as pd
from collections import Counter
import sys

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8') 

def analyze_performance():
    print("--- Recent Performance Analysis (Last 50 Signals) ---")
    try:
        conn = sqlite3.connect("signals.db")
        cursor = conn.cursor()
        
        # Query recent COMPLETED trades (SUCCESS/FAIL)
        # Note: Outcome might be NULL in some cases if db structure varies, but we filter for known states
        query = """
        SELECT symbol, signal, outcome, timestamp, confidence 
        FROM signals 
        WHERE outcome IN ('SUCCESS', 'FAIL') 
        ORDER BY timestamp DESC 
        LIMIT 50
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("No completed trades found in recent history.")
            conn.close()
            return

        cols = ['Symbol', 'Type', 'Outcome', 'Time', 'Conf']
        df = pd.DataFrame(rows, columns=cols)
        
        total = len(df)
        wins = len(df[df['Outcome'] == 'SUCCESS'])
        losses = len(df[df['Outcome'] == 'FAIL'])
        win_rate = (wins / total * 100) if total > 0 else 0
        
        print(f"\nTotal Completed: {total}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        print("\n--- Recent Trades ---")
        # Print simple table
        print(df[['Symbol', 'Type', 'Outcome', 'Time']].head(10).to_string(index=False))
        
        print("\n--- Breakdown by Pair ---")
        pair_counts = df.groupby('Symbol')['Outcome'].value_counts().unstack().fillna(0)
        print(pair_counts)
        
        # Check JPY specific
        print("\n--- JPY Analysis ---")
        jpy_trades = df[df['Symbol'].str.endswith('JPY')]
        if not jpy_trades.empty:
            jpy_wins = len(jpy_trades[jpy_trades['Outcome'] == 'SUCCESS'])
            jpy_total = len(jpy_trades)
            if jpy_total > 0:
                print(f"JPY Pairs Win Rate: {(jpy_wins/jpy_total*100):.1f}% ({jpy_wins}/{jpy_total})")
        else:
            print("No JPY trades found.")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        analyze_performance()
    except Exception as e:
        print(f"Script Error: {e}")
