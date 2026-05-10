import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
query = """
    SELECT id, symbol, signal, outcome, status, is_hidden, confidence, timestamp, regime, expert_signal 
    FROM signals 
    WHERE symbol = 'AUDUSD' AND (signal = 'BUY' OR expert_signal = 'BUY')
    ORDER BY timestamp DESC
    LIMIT 100
"""
df = pd.read_sql(query, conn)
print(df[['id', 'signal', 'expert_signal', 'status', 'is_hidden', 'timestamp']])
conn.close()
