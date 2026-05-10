import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
df = pd.read_sql_query("SELECT id, timestamp, symbol, signal, confidence, is_proven, is_hidden, status, mt5_ticket, outcome FROM signals WHERE id=141", conn)
print(df.to_string())
conn.close()
