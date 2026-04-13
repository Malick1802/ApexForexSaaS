import os
import sys
import sqlite3
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core.inference import InferenceEngine

def verify_locking():
    print("--- 🔒 Signal Locking Verification ---")
    engine = InferenceEngine()
    
    # Gold Scan
    # This should return the 'ACTIVE' BUY signal #25009 from the DB,
    # even if current market sentiment is neutral.
    res = engine.predict_symbol('GOLD', save_to_db=False, allow_stale=True)
    
    if res and res.get('is_locked'):
        print(f"✅ SUCCESS: GOLD is LOCKED.")
        print(f"   Signal: {res['signal']} (ID: {res.get('id')})")
        print(f"   Confidence: {res['confidence']:.1%}")
    else:
        print("❌ FAILURE: GOLD is NOT locked.")
        if res:
            print(f"   Returned Signal: {res['signal']}")

if __name__ == "__main__":
    verify_locking()
