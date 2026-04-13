import sys
import os
from pathlib import Path

# Fix path to root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from core.core.executive import ExecutiveEngine
from core.core.database import SignalDatabase
from datetime import datetime, timezone

def test_escalation():
    db = SignalDatabase()
    # Clean prev tests
    with db._get_connection() as conn:
        conn.execute("DELETE FROM signals WHERE symbol = 'EURAUD'")
    
    exec_eng = ExecutiveEngine()
    
    print("--- STEP 1: Creating 60% Benched Signal ---")
    res1 = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': 'EURAUD',
        'signal': 'WAIT',
        'expert_signal': 'BUY',
        'confidence': 0.61,
        'confidence_tier': 60,
        'price_at_signal': 1.6500,
        'tp_price': 1.6600,
        'sl_price': 1.6450,
        'outcome': 'ACTIVE',
        'is_hidden': 1,
        'is_proven': 0
    }
    db.save_signal(res1)
    
    print("--- STEP 2: Attempting 90% Escalation ---")
    res2 = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': 'EURAUD',
        'signal': 'WAIT',
        'expert_signal': 'BUY',
        'confidence': 0.91,
        'confidence_tier': 90,
        'price_at_signal': 1.6510,
        'tp_price': 1.6610,
        'sl_price': 1.6460,
        'outcome': 'ACTIVE',
        'is_hidden': 1,
        'is_proven': 0
    }
    
    # We use analyze_symbol proxy or just check db logic
    # Since analyze_symbol calls predict_symbol which fetches real data, 
    # we'll mock the active signal check.
    
    # Simulation of Executive Escalation Check
    # (Matches the logic added to core/executive.py)
    active = db.get_active_signals(symbol='EURAUD', include_hidden=True)
    print(f"Active signals before escalation: {len(active)} (Highest Tier: {max([s.get('confidence_tier', 0) for s in active], default=0)}%)")
    
    # Simulate Executive check from core/executive.py
    highest_active_tier = max([int(s.get('confidence_tier', 0)) for s in active], default=0)
    if res2['confidence_tier'] > highest_active_tier:
        print(f"🚀 Escalation detected: {res2['confidence_tier']}% > {highest_active_tier}%")
        db.save_signal(res2)
    else:
        print(f"⏭ No escalation ({res2['confidence_tier']}% <= {highest_active_tier}%)")

    # Final Check
    active_after = db.get_active_signals(symbol='EURAUD', include_hidden=True)
    print(f"Active signals after escalation: {len(active_after)}")
    for s in active_after:
        print(f"  - ID {s['id']}: Tier {s.get('confidence_tier', 0)}% ({s['signal']}/{s.get('expert_signal', 'WAIT')})")

if __name__ == "__main__":
    test_escalation()
