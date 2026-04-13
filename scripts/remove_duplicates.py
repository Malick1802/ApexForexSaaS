
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def remove_duplicates():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Fetch all signals
    query = "SELECT id, symbol, signal, outcome, timestamp FROM signals ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No signals found.")
        conn.close()
        return

    print(f"Total signals before deduplication: {len(df)}")

    # 2. Identify Duplicates
    # Criteria: Same Symbol, Same Signal Type, Created within 60 seconds of each other
    # We will sort by Symbol, Signal, Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['symbol', 'signal', 'timestamp'])
    
    ids_to_delete = []
    
    prev_row = None
    
    for index, row in df.iterrows():
        if prev_row is not None:
            # Check overlap
            same_symbol = (row['symbol'] == prev_row['symbol'])
            same_signal = (row['signal'] == prev_row['signal'])
            time_diff = (row['timestamp'] - prev_row['timestamp']).total_seconds()
            
            # If same symbol+signal and time diff < 120 seconds (2 mins), consider duplicate
            # Exception: If outcomes are different, maybe keep? But user said "Eliminate duplicates".
            # Usually we want to keep the one that is ACTIVE or has a later ID.
            
            if same_symbol and same_signal and abs(time_diff) < 120:
                print(f"Duplicate found: {row['symbol']} {row['signal']} at {row['timestamp']} (ID {row['id']}) vs {prev_row['timestamp']} (ID {prev_row['id']})")
                
                # Logic: Keep the one with the higher ID (latest inserted/updated)
                # But logic here iterates sorted by timestamp. 
                # If times are close, ID should be higher for the later one.
                # We mark the 'prev_row' (older) for deletion usually, or the one with "Worse" status?
                # Let's just keep the Latest ID.
                
                if row['id'] > prev_row['id']:
                    ids_to_delete.append(prev_row['id'])
                    prev_row = row # Move head to current
                else:
                    ids_to_delete.append(row['id'])
                    # prev_row stays same
            else:
                prev_row = row
        else:
            prev_row = row

    print(f"Found {len(ids_to_delete)} duplicates to delete.")
    
    if ids_to_delete:
        # Batch delete
        cursor.execute(f"DELETE FROM signals WHERE id IN ({','.join(['?']*len(ids_to_delete))})", ids_to_delete)
        conn.commit()
        print("Duplicates deleted.")
    
    # Verify
    count_after = cursor.execute("SELECT count(*) FROM signals").fetchone()[0]
    print(f"Total signals after deduplication: {count_after}")
    
    conn.close()

if __name__ == "__main__":
    remove_duplicates()
