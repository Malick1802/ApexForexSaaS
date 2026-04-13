
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def dedup_active_light():
    print(f"Connecting to DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Find pairs with > 1 active signal
        query = """
        SELECT symbol, signal, count(*) as count 
        FROM signals 
        WHERE outcome = 'ACTIVE' 
        GROUP BY symbol, signal 
        HAVING count > 1
        """
        rows = cursor.execute(query).fetchall()
        
        if not rows:
            print("No duplicate active signals found.")
            conn.close()
            return

        print(f"Found {len(rows)} symbol/signal pairs with duplicates.")
        
        deleted_total = 0
        for row in rows:
            sym = row[0]
            sig = row[1]
            
            # Get IDs ordered by timestamp DESC (latest first)
            q_ids = """
            SELECT id FROM signals 
            WHERE symbol = ? AND signal = ? AND outcome = 'ACTIVE' 
            ORDER BY timestamp DESC
            """
            ids = [r[0] for r in cursor.execute(q_ids, (sym, sig)).fetchall()]
            
            # Keep the first one (latest), delete the rest
            to_delete = ids[1:]
            if to_delete:
                placeholders = ','.join(['?']*len(to_delete))
                del_q = f"DELETE FROM signals WHERE id IN ({placeholders})"
                cursor.execute(del_q, to_delete)
                deleted_total += len(to_delete)
                # print(f"  {sym} {sig}: Kept ID {ids[0]}, deleted {len(to_delete)} duplicates")

        conn.commit()
        print(f"Cleanup Complete. Deleted {deleted_total} redundant active signals.")
        
        # Verify final count
        rem_q = "SELECT count(*) FROM signals WHERE outcome = 'ACTIVE'"
        remaining = cursor.execute(rem_q).fetchone()[0]
        print(f"Remaining Active Signals: {remaining}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dedup_active_light()
