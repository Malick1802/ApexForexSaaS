import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
query = """
    SELECT symbol, signal, outcome, confidence_tier, timestamp 
    FROM signals 
    WHERE symbol = 'AUDUSD' AND outcome IN ('SUCCESS', 'FAIL')
"""
df = pd.read_sql(query, conn)
print(df)
conn.close()
