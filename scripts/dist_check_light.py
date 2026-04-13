
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def check_distribution_light():
    print(f"Connecting to DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        print("--- Active Signals Breakdown ---")
        query = """
        SELECT outcome, signal, count(*) 
        FROM signals 
        GROUP BY outcome, signal
        """
        rows = cursor.execute(query).fetchall()
        
        for row in rows:
            print(f"Outcome: {row[0]:<10} Signal: {row[1]:<10} Count: {row[2]}")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_distribution_light()
