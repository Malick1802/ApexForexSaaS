import sqlite3

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print('=== ACTIVE SIGNALS (LIVE + SHADOW) ===')
cur.execute("SELECT timestamp, symbol, signal, confidence, raw_confidence, is_hidden, regime, outcome FROM signals WHERE outcome='ACTIVE' ORDER BY timestamp DESC")
rows = cur.fetchall()
if rows:
    for r in rows:
        tag = 'SHADOW' if r['is_hidden'] else 'LIVE'
        print(f"[{tag}] {r['symbol']:10} {r['signal']:5} conf={r['confidence']:.1%} raw={r['raw_confidence'] or 0:.1%} regime={str(r['regime'] or 'N/A'):20} ts={r['timestamp'][:16]}")
else:
    print('No active signals found.')

print()
print('=== RECENT BUY/SELL SIGNALS (last 24h) ===')
cur.execute("""SELECT symbol, signal, confidence, outcome, is_hidden, timestamp
              FROM signals
              WHERE signal IN ('BUY','SELL')
              AND timestamp >= datetime('now', '-24 hours')
              ORDER BY timestamp DESC LIMIT 30""")
rows = cur.fetchall()
if rows:
    for r in rows:
        tag = 'SHADOW' if r['is_hidden'] else 'LIVE'
        print(f"[{tag}] {r['symbol']:10} {r['signal']:5} {r['outcome']:8} conf={r['confidence']:.1%} ts={r['timestamp'][:16]}")
else:
    print('No BUY/SELL signals in last 24h.')

conn.close()
