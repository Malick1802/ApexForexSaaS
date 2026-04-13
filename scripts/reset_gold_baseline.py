import sqlite3
import json
from pathlib import Path

DB_PATH = Path("signals.db")
WL_PATH = Path("config/trading_whitelist.json")

# ── 1. Remove closed Gold signal (FAIL - no real outcome) ──────────────
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Delete the FAIL trade + all EXPIRED/WAIT signals for GOLD
# Keep only the currently ACTIVE BUY (ID 25039)
cursor.execute("""
    DELETE FROM signals
    WHERE symbol = 'GOLD'
    AND outcome != 'ACTIVE'
""")
deleted = cursor.rowcount
conn.commit()
conn.close()
print(f"✅ Deleted {deleted} closed/expired GOLD signals from DB.")

# Verify what remains
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT id, signal, outcome FROM signals WHERE symbol = 'GOLD'")
remaining = cursor.fetchall()
conn.close()
print(f"   Remaining GOLD signals: {[dict(r) for r in remaining]}")

# ── 2. Reset Gold performance matrix to zero ────────────────────────────
# The baseline was synthetic (Audit Baseline Bayesian), not real trades.
if WL_PATH.exists():
    data = json.load(open(WL_PATH))
    matrix = data.get("performance_matrix", {})
    
    if "GOLD" in matrix:
        for tier in ["60", "70", "80", "90", "100"]:
            matrix["GOLD"][tier] = {
                "alpha": 2.0,
                "beta": 2.0,
                "accuracy": 0.0,
                "trades": 0,
                "status": "BENCHED",
                "last_updated": "2026-04-06T16:12:00+00:00",
                "source": "Reset — awaiting real trade outcomes"
            }
        
        data["performance_matrix"] = matrix
        json.dump(data, open(WL_PATH, "w"), indent=2)
        print("✅ Gold performance matrix reset to zero (all tiers BENCHED).")
    else:
        print("❌ GOLD not found in matrix.")
else:
    print("❌ Whitelist not found.")

print("\nDone. Gold is now starting from a clean slate.")
print("The current ACTIVE BUY trade (ID 25039) is preserved.")
print("When it resolves, it will be the FIRST real certified data point.")
