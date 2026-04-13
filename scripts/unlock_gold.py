import sqlite3
from pathlib import Path

def clean_stale_signals():
    db_path = Path("signals.db")
    if not db_path.exists():
        print("signals.db not found")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Force expire any persistent 'WAIT' signals that are blocking the 'BUY'
        # Signal 25032 was the specifically identified blocking WAIT.
        cursor.execute("UPDATE signals SET outcome = 'EXPIRED' WHERE symbol = 'GOLD' AND signal = 'WAIT' AND outcome = 'ACTIVE'")
        conn.commit()
        
        print(f"✅ Success: Expired {cursor.rowcount} stale Gold WAIT signals.")
        conn.close()
    except Exception as e:
        print(f"❌ Error cleaning database: {e}")

if __name__ == "__main__":
    clean_stale_signals()
