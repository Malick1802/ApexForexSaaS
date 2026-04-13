import sqlite3

def check_new_trades():
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check all BUY or SELL signals
    cursor.execute("""
        SELECT id, timestamp, symbol, signal, confidence, outcome, mt5_ticket 
        FROM signals 
        WHERE signal IN ('BUY', 'SELL') 
        ORDER BY timestamp DESC LIMIT 10
    """)
    
    print(f"{'ID':<6} {'Pair':<10} {'Action':<6} {'Conf':<6} {'Status':<10} {'Ticket':<10} {'Time'}")
    print("-" * 75)
    for row in cursor.fetchall():
        conf = f"{row['confidence']*100:.1f}%" if row['confidence'] else "N/A"
        ticket = row['mt5_ticket'] if row['mt5_ticket'] is not None else "PENDING"
        ts = row['timestamp'][:19].replace("T", " ")
        print(f"{row['id']:<6} {row['symbol']:<10} {row['signal']:<6} {conf:<6} {row['outcome']:<10} {ticket:<10} {ts}")
        
if __name__ == '__main__':
    check_new_trades()
