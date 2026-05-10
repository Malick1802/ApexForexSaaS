import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

conn = sqlite3.connect('signals.db')
query = "SELECT id, symbol, signal, is_hidden, confidence, outcome, timestamp FROM signals WHERE signal='SELL' AND outcome='FAIL' AND is_hidden=0 ORDER BY id DESC LIMIT 15"
try:
    df = pd.read_sql_query(query, conn)
    print("LIVE FAILS:")
    print(df)
except Exception as e:
    print(e)
finally:
    conn.close()
