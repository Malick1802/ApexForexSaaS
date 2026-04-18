import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def generate_full_matrix():
    db_path = 'signals.db'
    conn = sqlite3.connect(db_path)
    
    # Total History
    df = pd.read_sql_query("SELECT * FROM signals", conn)
    
    # Filter for resolved trades only
    resolved = df[df['outcome'].isin(['SUCCESS', 'FAIL'])]
    
    stats = {
        "Total Signals Generated": len(df),
        "Total Resolved Trades": len(resolved),
        "Overall Win Rate": (len(resolved[resolved['outcome'] == 'SUCCESS']) / len(resolved) * 100) if len(resolved) > 0 else 0,
        "Active Trades": len(df[df['outcome'] == 'ACTIVE']),
        "Shadow Trades": len(df[df['is_hidden'] == 1]),
        "Certified Trades": len(df[df['is_hidden'] == 0])
    }
    
    # Per Symbol Win Rate (High Volume Only)
    symbol_stats = []
    if not resolved.empty:
        for symbol, group in resolved.groupby('symbol'):
            wins = len(group[group['outcome'] == 'SUCCESS'])
            total = len(group)
            wr = (wins / total) * 100
            symbol_stats.append({'Symbol': symbol, 'Trades': total, 'WinRate': f"{wr:.1f}%"})
    
    # Recent Window (Last 7 Days)
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    recent = resolved[resolved['timestamp'] >= cutoff]
    recent_wr = (len(recent[recent['outcome'] == 'SUCCESS']) / len(recent) * 100) if not recent.empty else 0
    
    print("--- PERFORMANCE SUMMARY ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    print("\n--- SYMBOL PERFORMANCE (Resolved) ---")
    symbol_df = pd.DataFrame(symbol_stats)
    if not symbol_df.empty:
        print(symbol_df.sort_values(by='Trades', ascending=False).to_string(index=False))
    
    print(f"\n7-Day Rolling Win Rate: {recent_wr:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    generate_full_matrix()
