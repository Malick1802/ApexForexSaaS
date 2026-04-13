
import sqlite3
import pandas as pd

def inspect_nzdjpy():
    try:
        conn = sqlite3.connect("signals.db")
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute("PRAGMA table_info(signals)")
        cols = [info[1] for info in cursor.fetchall()]
        
        # Query NZDJPY
        cursor.execute("SELECT * FROM signals WHERE symbol='NZDJPY' ORDER BY timestamp DESC LIMIT 5")
        rows = cursor.fetchall()
        
        print(f"--- NZDJPY Inspection ({len(rows)} records) ---")
        if not rows:
            print("No records found for NZDJPY.")
            return

        for row in rows:
            data = dict(zip(cols, row))
            print("\n------------------------------------------------")
            print(f"ID: {data.get('id')} | Time: {data.get('timestamp')}")
            print(f"Signal: {data.get('signal')} | Conf: {data.get('confidence')}")
            print(f"Entry: {data.get('price_at_signal')} | Current Outcome: {data.get('outcome')}")
            print(f"TP: {data.get('tp_price')} | SL: {data.get('sl_price')}")
            print(f"Model ID: {data.get('model_version')}")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_nzdjpy()
