import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 1. Find ALL shadow signals from today (including EXPIRED ones we accidentally cleared)
today = '2026-04-07'
cursor.execute("""
    SELECT id, symbol, signal, outcome, is_hidden, confidence_tier, timestamp
    FROM signals
    WHERE date(timestamp) = ? AND signal IN ('BUY','SELL')
    ORDER BY timestamp DESC
""", (today,))
rows = cursor.fetchall()
print(f"=== Today's BUY/SELL signals (including expired): {len(rows)} ===")
for r in rows:
    print(f"  ID={r['id']} | {r['symbol']} {r['signal']} | {r['outcome']} | hidden={r['is_hidden']} | tier={r['confidence_tier']} | {r['timestamp'][:16]}")

# 2. Count how many signals with outcome=EXPIRED were actually BUY/SELL before our cleanup
cursor.execute("""
    SELECT symbol, signal, COUNT(*) as n FROM signals
    WHERE outcome='EXPIRED' AND signal IN ('BUY','SELL')
    GROUP BY symbol, signal
    ORDER BY n DESC
""")
rows2 = cursor.fetchall()
print(f"\n=== Accidentally expired BUY/SELL signals (if any): {len(rows2)} groups ===")
for r in rows2:
    print(f"  {r['symbol']} {r['signal']}: {r['n']} signals")

# 3. Check telegram config
print("\n=== Checking Telegram config ===")
try:
    import yaml
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    tg = cfg.get('telegram', {})
    token = tg.get('bot_token', 'NOT SET')
    chat_id = tg.get('chat_id', 'NOT SET')
    print(f"  Token: {'SET (' + token[:10] + '...)' if token and token != 'NOT SET' else 'NOT SET'}")
    print(f"  Chat ID: {chat_id}")
except Exception as e:
    print(f"  Config error: {e}")

conn.close()
