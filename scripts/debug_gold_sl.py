import sqlite3

def check_gold():
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol, signal, price_at_signal, sl_price, tp_price, outcome, timestamp FROM signals WHERE symbol = 'GOLD' AND outcome = 'ACTIVE'")
    rows = cursor.fetchall()
    
    if not rows:
        print("No active GOLD signals found.")
    else:
        for r in rows:
            print(dict(r))

    # Also check the last 10 GOLD signals regardless of outcome
    print("\n--- Last 10 GOLD signals ---")
    cursor.execute("SELECT id, symbol, signal, price_at_signal, sl_price, tp_price, outcome, timestamp FROM signals WHERE symbol = 'GOLD' ORDER BY timestamp DESC LIMIT 10")
    [print(dict(r)) for r in cursor.fetchall()]

if __name__ == '__main__':
    check_gold()
