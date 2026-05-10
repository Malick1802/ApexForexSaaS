import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
query = "SELECT id, symbol, signal, confidence, notified, status, timestamp FROM signals WHERE symbol='GBPCHF' ORDER BY id DESC LIMIT 5"
try:
    df = pd.read_sql_query(query, conn)
    print(df)
except Exception as e:
    print(e)
finally:
    conn.close()
