import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta

# Project Root Resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "signals.db")

def report_trades_since_monday():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    # Today is 2026-05-04 (Monday). Last Monday was 2026-04-27.
    last_monday = "2026-04-27T00:00:00"
    
    conn = sqlite3.connect(DB_PATH)
    
    # Query for resolved signals using the correct outcome vocabulary
    query = f"""
        SELECT 
            symbol, 
            signal, 
            confidence, 
            status, 
            outcome, 
            tp_pips,
            sl_pips,
            timestamp
        FROM signals 
        WHERE timestamp >= '{last_monday}'
        AND outcome IN ('SUCCESS', 'FAIL')
        ORDER BY timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn)
    
    print(f"--- Trade Performance Report (Since Last Monday: 2026-04-27) ---")
    if len(df) == 0:
        print("No resolved trades found in the specified period.")
    else:
        # Calculate Pips based on outcome (SUCCESS = WIN, FAIL = LOSS)
        df['realized_pips'] = df.apply(lambda x: x['tp_pips'] if x['outcome'] == 'SUCCESS' else -x['sl_pips'], axis=1)
        
        # Calculate Stats
        wins = len(df[df['outcome'] == 'SUCCESS'])
        losses = len(df[df['outcome'] == 'FAIL'])
        total = wins + losses
        win_rate = (wins / total) * 100 if total > 0 else 0
        total_pips = df['realized_pips'].sum()
        
        print(f"Total Resolved Trades: {total}")
        print(f"Wins (SUCCESS): {wins}")
        print(f"Losses (FAIL): {losses}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Pips: {total_pips:+.1f}")
        
        print("\n--- Detailed Trade Log ---")
        # Format confidence to percentage
        df['confidence'] = (df['confidence'] * 100).round(1).astype(str) + "%"
        # Select columns for display
        display_df = df[['timestamp', 'symbol', 'signal', 'confidence', 'outcome', 'realized_pips']]
        print(display_df.to_string(index=False))

    # Also check ACTIVE trades
    active_query = f"""
        SELECT symbol, signal, confidence, timestamp, status
        FROM signals
        WHERE timestamp >= '{last_monday}'
        AND (status = 'ACTIVE' OR (status = 'NEW' AND is_hidden = 0))
        ORDER BY timestamp DESC
    """
    df_active = pd.read_sql_query(active_query, conn)
    if len(df_active) > 0:
        print("\n--- Currently Active/Pending Live Trades ---")
        df_active['confidence'] = (df_active['confidence'] * 100).round(1).astype(str) + "%"
        print(df_active.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    report_trades_since_monday()
