import sqlite3
import os

dbs = ['signals.db', 'data/signals.db']

for db_path in dbs:
    if not os.path.exists(db_path):
        print(f"Skipping {db_path} - not found")
        continue
        
    print(f"Migrating {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if outcome exists
        try:
            cursor.execute("SELECT outcome FROM signals LIMIT 1")
            print("  Outcome column already exists")
        except sqlite3.OperationalError:
            print("  Adding outcome column...")
            cursor.execute("ALTER TABLE signals ADD COLUMN outcome TEXT DEFAULT 'ACTIVE'")
            
        # Also ensure raw_probabilities, tp_price, sl_price exist (common missing ones)
        for col in ['raw_probabilities TEXT', 'tp_price REAL', 'sl_price REAL']:
            col_name = col.split()[0]
            try:
                cursor.execute(f"SELECT {col_name} FROM signals LIMIT 1")
            except sqlite3.OperationalError:
                print(f"  Adding {col_name} column...")
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {col}")
                
        conn.commit()
        conn.close()
        print(f"Successfully migrated {db_path}")
    except Exception as e:
        print(f"Error migrating {db_path}: {e}")

print("Done")
