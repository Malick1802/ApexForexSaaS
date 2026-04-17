import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'signals.db'

def cleanup():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find groups of potential duplicates
    query = """
    SELECT symbol, signal, confidence_tier, is_hidden, COUNT(*) as count
    FROM signals
    WHERE outcome IN ('ACTIVE', 'N/A')
    GROUP BY symbol, signal, confidence_tier, is_hidden
    HAVING count > 1
    """
    
    dupe_groups = cursor.execute(query).fetchall()
    
    if not dupe_groups:
        print("No duplicate groups found.")
        conn.close()
        return

    print(f"Found {len(dupe_groups)} duplicate groups. Cleaning...")
    
    total_superseded = 0
    for group in dupe_groups:
        # Get all IDs in this group, ordered by timestamp desc (keep the NEWEST)
        rows = cursor.execute("""
            SELECT id FROM signals 
            WHERE symbol = ? AND signal = ? AND confidence_tier = ? AND is_hidden = ?
            AND outcome IN ('ACTIVE', 'N/A')
            ORDER BY timestamp DESC
        """, (group['symbol'], group['signal'], group['confidence_tier'], group['is_hidden'])).fetchall()
        
        # Keep the first one (newest), supersede the rest
        ids_to_supersede = [row['id'] for row in rows[1:]]
        
        for sig_id in ids_to_supersede:
            cursor.execute("UPDATE signals SET outcome = 'SUPERSEDED' WHERE id = ?", (sig_id,))
            total_superseded += 1

    conn.commit()
    conn.close()
    print(f"Cleanup complete. Marked {total_superseded} signals as SUPERSEDED.")

if __name__ == "__main__":
    cleanup()
