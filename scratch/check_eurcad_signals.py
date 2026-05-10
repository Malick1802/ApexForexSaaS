import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
query = "SELECT * FROM signals WHERE symbol='EURCAD' ORDER BY timestamp DESC LIMIT 5"
df = pd.read_sql_query(query, conn)
print(df.to_string())
conn.close()
