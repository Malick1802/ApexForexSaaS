import sqlite3
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
db_path = PROJECT_ROOT / "signals.db"

def check_db():
    if not db_path.exists():
        print("Database not found.")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT symbol, signal, confidence, buy_prob, sell_prob, timestamp FROM signals ORDER BY timestamp DESC LIMIT 20")
        rows = cur.fetchall()
        
        if not rows:
            print("No signals found in database.")
        else:
            print(f"{'Symbol':<10} | {'Signal':<8} | {'Conf':<6} | {'BuyP':<6} | {'SellP':<6} | {'Time'}")
            print("-" * 70)
            for r in rows:
                sym, sig, conf, bp, sp, ts = r
                conf_str = f"{conf:.2f}" if conf is not None else "N/A"
                bp_str = f"{bp:.2f}" if bp is not None else "N/A"
                sp_str = f"{sp:.2f}" if sp is not None else "N/A"
                print(f"{sym:<10} | {sig:<8} | {conf_str:<6} | {bp_str:<6} | {sp_str:<6} | {ts}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_db()
