import sqlite3
conn = sqlite3.connect('signals.db')
cursor = conn.cursor()
cursor.execute('SELECT id, timestamp, signal, outcome, sl_price, tp_price FROM signals WHERE symbol="GOLD" ORDER BY timestamp DESC LIMIT 5')
for row in cursor.fetchall():
    print(row)
conn.close()
