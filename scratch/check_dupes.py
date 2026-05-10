import sqlite3

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT timestamp, symbol, signal, confidence_tier, is_hidden FROM signals WHERE signal IN ('BUY', 'SELL') ORDER BY timestamp DESC LIMIT 200")
rows = [dict(row) for row in cursor.fetchall()]

for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        r1 = rows[i]
        r2 = rows[j]
        if r1['symbol'] == r2['symbol'] and r1['signal'] == r2['signal'] and r1['confidence_tier'] == r2['confidence_tier']:
            # Check if they are close in time (string comparison for first 19 chars: YYYY-MM-DDTHH:MM:SS)
            if r1['timestamp'][:19] == r2['timestamp'][:19]:
                print(f"DUPE FOUND: {r1['symbol']} {r1['signal']} {r1['confidence_tier']}% at {r1['timestamp']}")

conn.close()
