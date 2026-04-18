import sqlite3
import os

db_path = 'signals.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Target Signal ID: 26425 (AUDUSD)
    # Restore it to SELL direction so it can be monitored/tracked correctly.
    print("Updating Signal ID 26425 (AUDUSD) to SELL status...")
    cursor.execute("""
        UPDATE signals 
        SET signal = 'SELL', 
            expert_signal = 'SELL',
            expert_intent = 'SELL'
        WHERE id = 26425
    """)
    
    if cursor.rowcount > 0:
        conn.commit()
        print("✅ Restore successful. AUDUSD is now a SELL signal.")
    else:
        print("❌ Signal ID 26425 not found in database.")
    
    conn.close()
else:
    print("❌ Database missing.")
