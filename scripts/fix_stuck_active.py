import sqlite3
from datetime import datetime, timedelta, timezone

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Expire the stuck EURGBP with no TP/SL resolution in 14+ days
cur.execute("UPDATE signals SET outcome = 'EXPIRED' WHERE id = 28302")
print(f"Expired EURGBP SELL (ID 28302): {cur.rowcount} row(s) updated")

# Show any new Gold/CrudeOil signals generated in last 24h
cur.execute("""
    SELECT id, symbol, signal, confidence, outcome, timestamp, tp_price, sl_price
    FROM signals
    WHERE symbol IN ('GOLD','XAUUSD','CrudeOIL','USOIL')
    AND signal IN ('BUY','SELL')
    AND datetime(timestamp) > datetime('now','-24 hours')
    ORDER BY timestamp DESC
""")
rows = cur.fetchall()
print()
print("--- GOLD/CRUDEOIL SIGNALS LAST 24H ---")
if not rows:
    print("None found")
for r in rows:
    print(r['id'], r['symbol'], r['signal'], r['outcome'], r['timestamp'][:16], "TP:", r['tp_price'], "SL:", r['sl_price'])

print()
print("--- ALL REMAINING ACTIVE BUY/SELL ---")
cur.execute("SELECT id, symbol, signal, outcome, timestamp FROM signals WHERE outcome='ACTIVE' AND signal IN ('BUY','SELL') ORDER BY timestamp DESC")
rows2 = cur.fetchall()
if not rows2:
    print("NONE - database is clean!")
for r in rows2:
    print(r['id'], r['symbol'], r['signal'], r['outcome'], r['timestamp'][:16])

conn.commit()
conn.close()
