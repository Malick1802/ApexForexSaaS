
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def add_columns():
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if columns exist
    cursor.execute("PRAGMA table_info(signals)")
    cols = [row[1] for row in cursor.fetchall()]
    
    if 'exit_price' not in cols:
        print("Adding 'exit_price' column...")
        cursor.execute("ALTER TABLE signals ADD COLUMN exit_price REAL")
    else:
        print("'exit_price' already exists.")
        
    if 'exit_reason' not in cols:
        print("Adding 'exit_reason' column...")
        cursor.execute("ALTER TABLE signals ADD COLUMN exit_reason TEXT")
    else:
        print("'exit_reason' already exists.")
        
    conn.commit()
    conn.close()
    print("Schema update complete.")

if __name__ == "__main__":
    add_columns()
