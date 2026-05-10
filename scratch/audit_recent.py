import sqlite3
conn = sqlite3.connect('signals.db')
cursor = conn.cursor()
cursor.execute("SELECT symbol, signal, outcome, timestamp FROM signals ORDER BY timestamp DESC LIMIT 20")
for row in cursor.fetchall():
    print(row)
conn.close()
