
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import SignalDatabase
from datetime import datetime, timezone

def audit_signals():
    db = SignalDatabase()
    active = db.get_active_signals(include_hidden=True)
    
    print(f"\n--- ACTIVE SIGNAL AUDIT ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC) ---")
    print(f"Total Active Signals: {len(active)}")
    
    if not active:
        print("No active signals found in DB.")
        return

    print(f"{'Symbol':<10} | {'Signal':<6} | {'Type':<8} | {'Tier':<4} | {'Outcome':<8} | {'Status':<10}")
    print("-" * 60)
    
    for s in active:
        is_shadow = bool(s.get('is_hidden', 0))
        sig_type = "SHADOW" if is_shadow else "LIVE"
        print(f"{s['symbol']:<10} | {s['signal']:<6} | {sig_type:<8} | {s.get('confidence_tier', '??'):<4} | {s.get('outcome', 'N/A'):<8} | {s.get('status', 'N/A'):<10}")

if __name__ == "__main__":
    audit_signals()
