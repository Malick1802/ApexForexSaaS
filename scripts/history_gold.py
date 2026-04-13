import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
df = pd.read_sql_query('SELECT id, timestamp, signal, outcome, sl_price, tp_price FROM signals WHERE symbol="GOLD" ORDER BY timestamp DESC LIMIT 10', conn)
print("=== Last 10 GOLD signals ===")
print(df.to_string())
conn.close()
