import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

db_path = "c:/Users/artem/Downloads/ApexForexSaaS/signals.db"
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT symbol, signal, expert_intent, round(buy_prob,3) as buy_prob, round(sell_prob,3) as sell_prob, round(wait_prob,3) as wait_prob, round(confidence,3) as confidence, timestamp FROM signals ORDER BY timestamp DESC LIMIT 15;", conn)
print(df)
conn.close()
