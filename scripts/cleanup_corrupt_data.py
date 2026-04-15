import sqlite3
import os
import json
from pathlib import Path
from core.database import SignalDatabase
from core.performance_gate import PerformanceGate

def cleanup():
    print("=== ApexForex SYSTEM PURGE ===")
    
    # 1. Connect to Database
    db_path = "signals.db"
    if not os.path.exists(db_path):
        print("Error: signals.db not found.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 2. Delete GOLD entries (wrong prices and corrupt volume)
        print("Cleaning up GOLD records...")
        cursor.execute("DELETE FROM signals WHERE symbol = 'GOLD'")
        cursor.execute("DELETE FROM signals WHERE symbol = 'XAUUSD'")
        print(f"  Rows deleted: {cursor.rowcount}")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Database Cleanup Error: {e}")

    # 3. Reset Gold in Performance Matrix
    print("Resetting Performance Whitelist...")
    try:
        gate = PerformanceGate()
        # Remove Gold from matrix to clear the 833k trade count
        if "GOLD" in gate.performance_matrix:
            del gate.performance_matrix["GOLD"]
            print("  Removed 'GOLD' from matrix.")
        if "XAUUSD" in gate.performance_matrix:
            del gate.performance_matrix["XAUUSD"]
            print("  Removed 'XAUUSD' from matrix.")
            
        gate.save_whitelist()
        print("  Whitelist saved.")
        
        # Recompute from healthy data (last 14 days)
        print("Recomputing healthy baseline...")
        gate.recompute_from_db(lookback_days=14)
        print("  Recompute complete.")
        
    except Exception as e:
        print(f"Whitelist Reset Error: {e}")

    print("\n[SUCCESS] System Purge Complete. You can now restart the Sentinel.")

if __name__ == "__main__":
    cleanup()
