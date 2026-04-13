import sqlite3
import json

def check_euraud():
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check recent EURAUD signals
    cursor.execute("""
        SELECT * FROM signals 
        WHERE symbol = 'EURAUD' 
        ORDER BY timestamp DESC LIMIT 20
    """)
    
    rows = cursor.fetchall()
    if not rows:
        print("No EURAUD signals found in the database.")
        return
        
    print(f"{'ID':<6} {'Signal':<6} {'Conf':<6} {'Status':<10} {'Outcome':<10} {'Timestamp'}")
    print("-" * 70)
    for row in rows:
        conf = f"{row['confidence']*100:.1f}%" if row['confidence'] else "N/A"
        raw_conf = f"{row['raw_confidence']*100:.1f}%" if 'raw_confidence' in row.keys() and row['raw_confidence'] else "N/A"
        print(f"{row['id']:<6} {row['signal']:<6} {conf:<6} {row['status']:<10} {row['outcome']:<10} {row['timestamp']}")

if __name__ == '__main__':
    check_euraud()
