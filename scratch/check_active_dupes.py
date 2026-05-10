import sqlite3

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all signals currently active
cursor.execute("SELECT timestamp, symbol, signal, confidence_tier, outcome FROM signals WHERE outcome IN ('ACTIVE', 'N/A') AND signal IN ('BUY', 'SELL')")
rows = [dict(row) for row in cursor.fetchall()]

dupes = {}
for r in rows:
    key = (r['symbol'], r['signal'], r['confidence_tier'])
    if key not in dupes:
        dupes[key] = []
    dupes[key].append(r['timestamp'])

found = False
for key, timestamps in dupes.items():
    if len(timestamps) > 1:
        found = True
        print(f"ACTIVE DUPLICATE: {key[0]} {key[1]} {key[2]}%")
        for ts in timestamps:
            print(f"  - {ts}")

if not found:
    print("No active duplicates found.")

conn.close()
