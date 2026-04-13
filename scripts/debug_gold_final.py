import os
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from core.inference import InferenceEngine

def debug_gold():
    print("--- 🔍 Debugging Gold Signal Resolution ---")
    engine = InferenceEngine()
    
    # 1. Database Check
    db_path = Path("signals.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol, signal, outcome FROM signals WHERE symbol = 'GOLD' AND outcome = 'ACTIVE'")
    active = cursor.fetchall()
    print(f"Active GOLD Signals in DB: {active}")
    conn.close()

    # 2. Prediction Check
    print("\nRunning predict_symbol('GOLD')...")
    res = engine.predict_symbol('GOLD', save_to_db=True, allow_stale=True)
    if res:
        print(f"SUCCESS: Signal={res.get('signal')}, Conf={res.get('confidence', 0)*100:.1f}%")
        print(f"Locked: {res.get('is_locked', False)}")
    else:
        print("FAILURE: predict_symbol returned None.")

if __name__ == "__main__":
    debug_gold()
