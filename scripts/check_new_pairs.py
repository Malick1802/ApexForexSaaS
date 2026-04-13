import sqlite3, sys
sys.path.insert(0, '.')
conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row

for sym in ['GOLD', 'CrudeOIL', 'USDSGD']:
    row = conn.execute('SELECT COUNT(*) as cnt FROM signals WHERE symbol=?', (sym,)).fetchone()
    cnt = row['cnt']
    latest = conn.execute(
        'SELECT timestamp, signal, confidence, outcome FROM signals WHERE symbol=? ORDER BY timestamp DESC LIMIT 1',
        (sym,)
    ).fetchone()
    if latest:
        sig = latest['signal']
        conf = float(latest['confidence']) * 100
        out = latest['outcome']
        ts = latest['timestamp'][:19]
        print(sym, ': total=', cnt, '| Latest:', sig, round(conf,1), '%', '['+out+']', '@', ts)
    else:
        print(sym, ': NO SIGNALS IN DB')

conn.close()
