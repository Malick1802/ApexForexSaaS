"""
Fresh Start Reset
=================
Clears all signal records from the databases and resets calibration/whitelist
state so the Foundation Brain can build a clean track record from zero.
"""
import sys
import sqlite3
import json
import shutil
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

DATABASES = [
    PROJECT_ROOT / "signals.db",
    PROJECT_ROOT / "signals_vm.db",
    PROJECT_ROOT / "data" / "signals.db",
]

WHITELIST_PATH = PROJECT_ROOT / "config" / "trading_whitelist.json"
WFA_RESULTS    = PROJECT_ROOT / "logs" / "wfa_expert_14d_results.json"

TABLES_TO_PURGE = ["signals", "audit_logs"]

print("\n" + "="*60)
print("  APEX FRESH START — SIGNAL & CALIBRATION RESET")
print("="*60 + "\n")

# ── 1. Purge database tables ─────────────────────────────────
for db_path in DATABASES:
    if not db_path.exists():
        print(f"  [SKIP] {db_path.name} — not found")
        continue

    # Back up first
    bak = db_path.with_suffix(f".{NOW}.bak")
    shutil.copy2(db_path, bak)
    print(f"  [BAK]  {db_path.name} → {bak.name}")

    try:
        conn = sqlite3.connect(str(db_path))
        cur  = conn.cursor()
        for table in TABLES_TO_PURGE:
            cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cur.fetchone():
                cur.execute(f"DELETE FROM {table}")
                conn.commit()
                print(f"  [PURGE] {db_path.name} → table '{table}' cleared")
        # Reset auto-increment counters
        cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('signals','audit_logs')")
        conn.commit()
        conn.close()
        print(f"  [OK]   {db_path.name} — all signals deleted\n")
    except Exception as e:
        print(f"  [ERR]  {db_path.name}: {e}\n")

# ── 2. Reset trading whitelist to blank slate ─────────────────
if WHITELIST_PATH.exists():
    shutil.copy2(WHITELIST_PATH, WHITELIST_PATH.with_suffix(f".{NOW}.bak"))
    print(f"  [BAK]  trading_whitelist.json → .{NOW}.bak")
    blank_whitelist = {}
    with open(WHITELIST_PATH, 'w') as f:
        json.dump(blank_whitelist, f, indent=2)
    print("  [RESET] trading_whitelist.json — cleared to empty dict\n")
else:
    print("  [SKIP] trading_whitelist.json — not found\n")

# ── 3. Clear stale WFA results ────────────────────────────────
if WFA_RESULTS.exists():
    WFA_RESULTS.unlink()
    print("  [DEL]  wfa_expert_14d_results.json — removed\n")

# ── 4. Summary ────────────────────────────────────────────────
print("="*60)
print("  RESET COMPLETE — System is clean.")
print("  The Foundation Brain will now build a fresh track record.")
print("  Restart the fleet: run RESTART.bat")
print("="*60 + "\n")
