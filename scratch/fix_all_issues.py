"""
Fix All Active Signal Issues:
1. Expire stale NZDJPY LIVE trade (3 days old, should have resolved)
2. Clean duplicate NZDUSD SHADOW signals (keep newest, expire oldest)
3. Diagnose CrudeOIL shadow issue
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = "signals.db"
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=" * 60)
print("ISSUE 1: NZDJPY Stale Live Trade")
print("=" * 60)
cur.execute("""
    SELECT id, timestamp, symbol, signal, confidence, outcome,
           tp_price, sl_price, price_at_signal
    FROM signals WHERE symbol='NZDJPY' AND outcome='ACTIVE'
""")
nzdjpy_rows = cur.fetchall()
for r in nzdjpy_rows:
    d = dict(r)
    print(f"  ID={d['id']} ts={d['timestamp'][:16]} signal={d['signal']} "
          f"conf={d['confidence']:.1%} entry={d['price_at_signal']} "
          f"TP={d['tp_price']} SL={d['sl_price']}")

if nzdjpy_rows:
    # Mark as EXPIRED — it has been open 3 days with no resolution
    ids = [r['id'] for r in nzdjpy_rows]
    for _id in ids:
        cur.execute("UPDATE signals SET outcome='EXPIRED' WHERE id=?", (_id,))
    conn.commit()
    print(f"  -> EXPIRED signal IDs: {ids}")
else:
    print("  -> No stale NZDJPY signals found. Already resolved.")


print()
print("=" * 60)
print("ISSUE 2: Duplicate NZDUSD SHADOW Signals")
print("=" * 60)
cur.execute("""
    SELECT id, timestamp, signal, confidence
    FROM signals
    WHERE symbol='NZDUSD' AND outcome='ACTIVE'
    ORDER BY timestamp DESC
""")
nzdusd_rows = cur.fetchall()
print(f"  Found {len(nzdusd_rows)} active NZDUSD signals:")
for r in nzdusd_rows:
    print(f"  ID={r['id']} ts={r['timestamp'][:16]} conf={r['confidence']:.1%}")

if len(nzdusd_rows) > 1:
    # Keep newest (index 0), expire the rest
    to_expire = [r['id'] for r in nzdusd_rows[1:]]
    for _id in to_expire:
        cur.execute("UPDATE signals SET outcome='EXPIRED' WHERE id=?", (_id,))
    conn.commit()
    print(f"  -> Expired duplicate IDs: {to_expire}")
    print(f"  -> Kept active: ID={nzdusd_rows[0]['id']}")
else:
    print("  -> No duplicates found.")


print()
print("=" * 60)
print("ISSUE 3: CrudeOIL Shadow-Only Despite Being Approved")
print("=" * 60)

# Check whitelist
wl_path = Path("config/trading_whitelist.json")
if wl_path.exists():
    whitelist = json.loads(wl_path.read_text())
    crude_entries = {k: v for k, v in whitelist.items() if "crude" in k.lower() or "crudes" in k.lower()}
    if crude_entries:
        print("  CrudeOIL in whitelist:")
        for k, v in crude_entries.items():
            print(f"    {k}: {v}")
    else:
        print("  CrudeOIL NOT FOUND in whitelist (case mismatch?)")
    # Show all keys
    print(f"  All whitelist keys: {list(whitelist.keys())[:10]}...")
else:
    print(f"  Whitelist not found at {wl_path}")

# Check model files
expert_dir = Path("models/expert/CrudeOIL")
specialist_dir = Path("models/specialist/CrudeOIL")
print(f"\n  models/expert/CrudeOIL exists: {expert_dir.exists()}")
if expert_dir.exists():
    print(f"    Files: {[f.name for f in expert_dir.iterdir()]}")
print(f"  models/specialist/CrudeOIL exists: {specialist_dir.exists()}")

# Show recent CrudeOIL signals
cur.execute("""
    SELECT id, timestamp, signal, confidence, is_hidden, outcome
    FROM signals WHERE symbol='CrudeOIL'
    ORDER BY timestamp DESC LIMIT 10
""")
crude_rows = cur.fetchall()
print(f"\n  Recent CrudeOIL signals:")
for r in crude_rows:
    tag = "SHADOW" if r['is_hidden'] else "LIVE"
    print(f"  [{tag}] ID={r['id']} ts={r['timestamp'][:16]} {r['signal']} "
          f"conf={r['confidence']:.1%} outcome={r['outcome']}")

conn.close()

print()
print("=" * 60)
print("DONE - Issues 1 & 2 fixed. Review Issue 3 output above.")
print("=" * 60)
