
import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def inspect_active_details():
    try:
        conn = sqlite3.connect(DB_PATH)
        # Fetch all columns for active signals
        query = "SELECT * FROM signals WHERE outcome='ACTIVE'"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No active signals.")
        else:
            print(f"Found {len(df)} Active Signals:")
            # Display key columns
            cols = ['id', 'symbol', 'signal', 'confidence', 'strategy', 'timestamp', 'price_at_signal']
            # Handle missing columns if any (e.g. strategy might be missing in some rows)
            available_cols = [c for c in cols if c in df.columns]
            print(df[available_cols].to_string(index=False))
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_active_details()
