import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone

conn = sqlite3.connect('signals.db')
cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()

query = """
    SELECT symbol, outcome, COUNT(*) as count 
    FROM signals 
    WHERE outcome IN ('SUCCESS', 'FAIL')
    GROUP BY symbol, outcome
"""
df = pd.read_sql(query, conn)
print(df)
conn.close()
