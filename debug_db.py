import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')
cursor = conn.cursor()
cursor.execute("SELECT symbol, signal, tp_price, sl_price, timestamp FROM signals WHERE symbol='GBPUSD' ORDER BY timestamp DESC LIMIT 1")
row = cursor.fetchone()
print(f"GBPUSD Record: {row}")

cursor.execute("SELECT symbol, signal, tp_price, sl_price, timestamp FROM signals WHERE symbol='AUDUSD' ORDER BY timestamp DESC LIMIT 1")
row2 = cursor.fetchone()
print(f"AUDUSD Record: {row2}")
