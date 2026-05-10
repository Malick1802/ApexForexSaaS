import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
query = """
    SELECT id, signal, outcome, confidence, timestamp, is_hidden
    FROM signals 
    WHERE symbol = 'GBPUSD' AND outcome IN ('SUCCESS', 'FAIL')
    ORDER BY timestamp DESC
"""
df = pd.read_sql(query, conn)
print(df)
conn.close()
