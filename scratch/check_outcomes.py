import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
query = """
    SELECT symbol, signal, outcome, confidence, timestamp, confidence_tier 
    FROM signals 
    WHERE outcome IN ('WIN', 'LOSS', 'FAIL') 
    AND symbol IN ('AUDUSD', 'EURUSD', 'GBPUSD', 'NZDJPY', 'EURAUD')
    ORDER BY timestamp DESC
"""
df = pd.read_sql(query, conn)
print(df.head(20))
conn.close()
