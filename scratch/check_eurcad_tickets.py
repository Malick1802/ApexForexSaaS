import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
df = pd.read_sql_query("SELECT id, timestamp, symbol, signal, status, mt5_ticket, outcome FROM signals WHERE symbol='EURCAD' ORDER BY timestamp DESC LIMIT 5", conn)
print(df.to_string())
conn.close()
