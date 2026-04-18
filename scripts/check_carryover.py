import sqlite3
import os

db_path = 'signals.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Find positions that are actually ACTIVE trades (not WAIT)
    cursor.execute("SELECT symbol, signal, confidence_tier, price_at_signal, timestamp FROM signals WHERE outcome = 'ACTIVE' AND signal IN ('BUY', 'SELL')")
    rows = cursor.fetchall()
    if rows:
        print("--- CARRY-OVER ACTIVE TRADES ---")
        for r in rows:
            print(f"{r['symbol']}: {r['signal']} @ {r['confidence_tier']}% (Entry: {r['price_at_signal']}) - {r['timestamp']}")
    else:
        print("No active BUY/SELL trades are currently in carry-over.")
    conn.close()
else:
    print("DB missing.")
