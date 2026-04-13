import sqlite3, sys
sys.path.insert(0, '.')
conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT id, timestamp, symbol, signal, confidence, outcome, buy_prob, sell_prob FROM signals WHERE symbol='CrudeOIL' ORDER BY timestamp DESC LIMIT 5").fetchall()
for r in rows:
    print(dict(r))
