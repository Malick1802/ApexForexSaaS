import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone

def check_shadow():
    conn = sqlite3.connect('signals.db')
    # We look for all non-WAIT signals that are hidden (Shadow)
    query = """
    SELECT symbol, signal, confidence, timestamp, is_hidden, outcome
    FROM signals 
    WHERE signal != 'WAIT' 
    AND (is_hidden = 1 OR outcome = 'N/A')
    ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No shadow signals found in the database.")
        return

    # Convert timestamp to datetime for filtering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Let's focus on the last 24 hours to see what's "current"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    df_recent = df[df['timestamp'] >= cutoff]
    
    print(f"Total Shadow Signals (Lifecycle): {len(df)}")
    print(f"Recent Shadow Signals (Last 24h): {len(df_recent)}")
    if not df_recent.empty:
        print("\n--- Recent Shadow Signal List ---")
        # Format for display
        display_df = df_recent[['symbol', 'signal', 'confidence', 'timestamp']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        print(display_df.to_string(index=False))

if __name__ == "__main__":
    check_shadow()
