
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from core.core.database import SignalDatabase
    print("✅ Imported SignalDatabase")
    
    # Initialize DB (triggers _init_db and migration)
    db = SignalDatabase()
    print("✅ Database Initialized (Migration should have run)")
    
    # Verify columns
    import sqlite3
    conn = sqlite3.connect('signals.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(signals)")
    cols = [row[1] for row in cursor.fetchall()]
    print(f"Current Columns: {cols}")
    
    if 'buy_prob' in cols:
        print("✅ SUCCESS: 'buy_prob' column exists.")
    else:
        print("❌ FAILURE: 'buy_prob' column MISSING.")
        
    conn.close()

except Exception as e:
    print(f"❌ Error: {e}")
