
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def deduplicate_active():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Find pairs with > 1 active signal
    query = """
    SELECT symbol, count(*) as count 
    FROM signals 
    WHERE outcome = 'ACTIVE' 
    GROUP BY symbol 
    HAVING count > 1
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No duplicate active signals found.")
        conn.close()
        return

    print(f"Found {len(df)} pairs with stacked active signals.")
    
    deleted_total = 0
    for sym in df['symbol']:
        # Get all active IDs for this symbol, ordered by timestamp DESC
        q_ids = """
        SELECT id FROM signals 
        WHERE symbol = ? AND outcome = 'ACTIVE' 
        ORDER BY timestamp DESC
        """
        ids = [row[0] for row in cursor.execute(q_ids, (sym,)).fetchall()]
        
        # Keep the first one (latest), delete the rest
        to_delete = ids[1:]
        if to_delete:
            cursor.execute(f"DELETE FROM signals WHERE id IN ({','.join(['?']*len(to_delete))})", to_delete)
            deleted_total += len(to_delete)
            # print(f"  {sym}: Kept ID {ids[0]}, deleted {len(to_delete)} duplicates")

    conn.commit()
    print(f"Cleanup Complete. Deleted {deleted_total} redundant active signals.")
    
    # Verify final count
    final_count = cursor.execute("SELECT count(*) FROM signals WHERE outcome = 'ACTIVE'").fetchone()[0]
    print(f"Remaining Active Signals: {final_count}")
    
    conn.close()

if __name__ == "__main__":
    deduplicate_active()
