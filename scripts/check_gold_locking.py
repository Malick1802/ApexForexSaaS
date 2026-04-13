import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check():
    db_path = "signals.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check Signal 25009
    cursor.execute("SELECT id, symbol, signal, outcome, timestamp FROM signals WHERE id = 25009")
    row = cursor.fetchone()
    if row:
        print(f"✅ Found Signal 25009: {row[1]} {row[2]} (Status: {row[3]}) at {row[4]}")
    else:
        print("❌ Signal 25009 NOT FOUND.")
        
    # 2. Check for ANY Active GOLD signals
    cursor.execute("SELECT id, symbol, signal, outcome FROM signals WHERE symbol = 'GOLD' AND outcome = 'ACTIVE'")
    active = cursor.fetchall()
    print(f"\nTotal Active GOLD Signals: {len(active)}")
    for r in active:
        print(f"  - Active ID {r[0]}: {r[2]}")
        
    conn.close()

if __name__ == "__main__":
    check()
