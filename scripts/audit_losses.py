import sqlite3
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== RECENT OUTCOMES (Last 7 Days) ===")
cur.execute("""
SELECT symbol, signal, confidence, outcome, timestamp, regime, is_hidden
FROM signals 
WHERE outcome IN ('SUCCESS','FAIL','EXPIRED')
AND datetime(timestamp) > datetime('now', '-7 days')
ORDER BY timestamp DESC
LIMIT 40
""")
rows = cur.fetchall()
for r in rows:
    ts = (r['timestamp'] or '')[:16]
    sym = (r['symbol'] or '').ljust(8)
    sig = (r['signal'] or '').ljust(4)
    conf = r['confidence'] or 0
    outcome = (r['outcome'] or '').ljust(7)
    regime = r['regime'] or 'N/A'
    shadow = r['is_hidden']
    print(f"{ts} | {sym} | {sig} | {conf:.0%} | {outcome} | Regime:{regime} | Shadow:{shadow}")

print()
print("=== WIN RATE SUMMARY (Last 7 Days) ===")
cur.execute("""
SELECT outcome, COUNT(*) as cnt
FROM signals
WHERE outcome IN ('SUCCESS','FAIL')
AND datetime(timestamp) > datetime('now', '-7 days')
GROUP BY outcome
""")
wins = 0
losses = 0
for r in cur.fetchall():
    print(f"  {r['outcome']}: {r['cnt']}")
    if r['outcome'] == 'SUCCESS': wins = r['cnt']
    if r['outcome'] == 'FAIL': losses = r['cnt']
total = wins + losses
if total > 0:
    print(f"  WIN RATE: {wins}/{total} = {wins/total:.1%}")

print()
print("=== PER-SYMBOL WIN RATE (Last 7 Days) ===")
cur.execute("""
SELECT symbol,
    SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN outcome='FAIL' THEN 1 ELSE 0 END) as losses,
    COUNT(*) as total
FROM signals
WHERE outcome IN ('SUCCESS','FAIL')
AND datetime(timestamp) > datetime('now', '-7 days')
GROUP BY symbol
ORDER BY total DESC
""")
for r in cur.fetchall():
    wr = r['wins'] / r['total'] if r['total'] > 0 else 0
    bar = '#' * r['wins'] + '.' * r['losses']
    print(f"  {(r['symbol'] or '').ljust(8)} | W:{r['wins']} L:{r['losses']} | WR:{wr:.0%} | {bar}")

print()
print("=== REGIME DISTRIBUTION (Last 7 Days, Failed Signals) ===")
cur.execute("""
SELECT regime, COUNT(*) as cnt
FROM signals
WHERE outcome = 'FAIL'
AND datetime(timestamp) > datetime('now', '-7 days')
GROUP BY regime
ORDER BY cnt DESC
""")
for r in cur.fetchall():
    print(f"  {r['regime']}: {r['cnt']}")

print()
print("=== ACTIVE SIGNALS ===")
cur.execute("""
SELECT symbol, signal, confidence, timestamp, regime, is_hidden
FROM signals WHERE outcome = 'ACTIVE' AND signal IN ('BUY','SELL')
ORDER BY timestamp DESC
""")
for r in cur.fetchall():
    ts = (r['timestamp'] or '')[:16]
    sym = (r['symbol'] or '').ljust(8)
    sig = (r['signal'] or '').ljust(4)
    conf = r['confidence'] or 0
    regime = r['regime'] or 'N/A'
    shadow = r['is_hidden']
    print(f"  {ts} | {sym} | {sig} | {conf:.0%} | Regime:{regime} | Shadow:{shadow}")

conn.close()
