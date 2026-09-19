"""
Promotion Gap Diagnostic
Identifies pairs that SHOULD be APPROVED but are not, and explains why.
"""
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "signals.db"
WHITELIST_PATH = PROJECT_ROOT / "config" / "trading_whitelist.json"

MIN_TRADES = 2
MIN_WIN_RATE = 0.70
LOOKBACK_DAYS = 14

def run():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"PROMOTION GAP DIAGNOSTIC  |  Cutoff: {cutoff[:10]}")
    print(f"{SEP}\n")

    # 1. Model versions present in last 14 days
    cur.execute(
        "SELECT DISTINCT model_version, COUNT(*) as cnt FROM signals "
        "WHERE timestamp >= ? GROUP BY model_version", (cutoff,)
    )
    rows = cur.fetchall()
    print("1. Model versions in last 14 days:")
    for r in rows:
        print(f"   [{r['model_version']}]  =>  {r['cnt']} signals")

    # 2. All qualifying pairs (ANY model version)
    cur.execute("""
        SELECT symbol, signal, model_version,
               SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END) AS wins,
               COUNT(*) AS total,
               ROUND(100.0*SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END)/COUNT(*),1) AS wr
        FROM signals
        WHERE outcome IN ('SUCCESS','FAIL')
          AND timestamp >= ?
        GROUP BY symbol, signal, model_version
        HAVING total >= ? AND wr >= ?
        ORDER BY wr DESC, total DESC
    """, (cutoff, MIN_TRADES, MIN_WIN_RATE * 100))
    all_qualifying = cur.fetchall()

    print(f"\n2. ALL pairs >={int(MIN_WIN_RATE*100)}% WR & >={MIN_TRADES} trades (any model version):")
    if not all_qualifying:
        print("   (none)")
    for r in all_qualifying:
        mv = r['model_version'] or 'None'
        print(f"   {r['symbol']:12} {r['signal']:4}  model={mv:20}  "
              f"{r['wins']}/{r['total']} ({r['wr']}%)")

    # 3. Qualifying pairs v1 ONLY (what the gate actually queries)
    cur.execute("""
        SELECT symbol, signal,
               SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END) AS wins,
               COUNT(*) AS total,
               ROUND(100.0*SUM(CASE WHEN outcome='SUCCESS' THEN 1 ELSE 0 END)/COUNT(*),1) AS wr
        FROM signals
        WHERE outcome IN ('SUCCESS','FAIL')
          AND model_version = 'v1'
          AND timestamp >= ?
        GROUP BY symbol, signal
        HAVING total >= ? AND wr >= ?
        ORDER BY wr DESC, total DESC
    """, (cutoff, MIN_TRADES, MIN_WIN_RATE * 100))
    v1_qualifying = cur.fetchall()

    print(f"\n3. Pairs >={int(MIN_WIN_RATE*100)}% WR & >={MIN_TRADES} trades (v1 ONLY -- what the gate sees):")
    if not v1_qualifying:
        print("   WARNING: NONE -- gate filters on model_version='v1' but no v1 signals qualify!")
    for r in v1_qualifying:
        print(f"   {r['symbol']:12} {r['signal']:4}  {r['wins']}/{r['total']} ({r['wr']}%)")

    # 4. All model_version values ever in DB
    cur.execute("SELECT DISTINCT model_version FROM signals ORDER BY model_version")
    all_versions = [r['model_version'] for r in cur.fetchall()]
    print(f"\n4. All model_version values ever stored in DB: {all_versions}")

    # 5. Whitelist APPROVED entries
    print(f"\n5. Whitelist cross-check (APPROVED entries):")
    if WHITELIST_PATH.exists():
        with open(WHITELIST_PATH) as f:
            wl = json.load(f)
        pm = wl.get("performance_matrix", {})
        approved = []
        for sym, dirs in pm.items():
            if sym == "SYSTEM":
                continue
            for direction, tiers in dirs.items():
                for tier, data in tiers.items():
                    if data.get("status") == "APPROVED":
                        approved.append((sym, direction, tier,
                                         data.get("accuracy", 0),
                                         data.get("trades", 0)))
        if approved:
            for sym, direction, tier, acc, trades in sorted(approved):
                print(f"   APPROVED  {sym:12} {direction:4}  tier={tier}  "
                      f"acc={acc:.0%}  trades={trades}")
        else:
            print("   WARNING: NO APPROVED entries exist in the whitelist!")
    else:
        print("   WARNING: Whitelist file not found!")

    # 6. Pairs that passed overall but are invisible to the gate (not v1)
    all_set = {(r['symbol'], r['signal']) for r in all_qualifying}
    v1_set  = {(r['symbol'], r['signal']) for r in v1_qualifying}
    missed  = all_set - v1_set

    print(f"\n6. Pairs qualified overall but INVISIBLE to the gate (wrong model_version):")
    if not missed:
        print("   (none -- all qualifying pairs are in v1 signals)")
    for sym, sig in sorted(missed):
        matches = [r for r in all_qualifying if r['symbol'] == sym and r['signal'] == sig]
        for r in matches:
            mv = r['model_version'] or 'None'
            print(f"   MISSED  {sym:12} {sig:4}  model={mv:20}  "
                  f"{r['wins']}/{r['total']} ({r['wr']}%)")

    # 7. Still-ACTIVE (unresolved) signals
    cur.execute("""
        SELECT symbol, signal, model_version, COUNT(*) as cnt
        FROM signals
        WHERE outcome = 'ACTIVE'
          AND timestamp >= ?
        GROUP BY symbol, signal, model_version
        ORDER BY cnt DESC
    """, (cutoff,))
    active = cur.fetchall()
    print(f"\n7. Still-ACTIVE (unresolved) signals in last 14d:")
    if not active:
        print("   (none)")
    for r in active:
        mv = r['model_version'] or 'None'
        print(f"   {r['symbol']:12} {r['signal']:4}  model={mv:20}  unresolved={r['cnt']}")

    conn.close()

    print(f"\n{SEP}")
    print("ROOT CAUSE SUMMARY")
    print(SEP)
    if not v1_qualifying and all_qualifying:
        print("!! CRITICAL: recompute_from_db() filters on model_version='v1'")
        print("   but the qualifying signals are stored under a different tag.")
        print("   Fix: remove the model_version='v1' filter from performance_gate.py")
        print("        OR ensure signals are saved with model_version='v1'.")
    elif not all_qualifying:
        print("INFO: No pairs have >= 2 resolved trades with >= 70% WR yet.")
        print("      System is still accumulating live history. No action needed.")
    else:
        print("OK: v1 signals are qualifying and promoting correctly.")
    print()

if __name__ == "__main__":
    run()
