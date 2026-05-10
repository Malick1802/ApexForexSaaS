import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
df = pd.read_sql_query("SELECT * FROM signals WHERE symbol='EURCAD' ORDER BY timestamp DESC LIMIT 1", conn)
print(df.columns.tolist())
conn.close()
