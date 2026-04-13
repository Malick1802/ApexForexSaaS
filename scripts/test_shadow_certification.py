import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import InferenceEngine
from core.database import SignalDatabase

def test_shadow_certification():
    db = SignalDatabase()
    engine = InferenceEngine(model_dir="models/expert")
    
    print("\n--- TEST 1: Shadow Certification Pair (EURAUD) ---")
    # EURAUD is currently BENCHED at the 60% tier.
    # It should generate a signal but mark it as HIDDEN.
    res_eu = engine.predict_symbol("EURAUD", save_to_db=True)
    
    if res_eu:
        sig_id = res_eu.get('id')
        is_hidden = res_eu.get('is_hidden')
        outcome = res_eu.get('outcome')
        signal = res_eu.get('signal')
        confidence = res_eu.get('confidence')
        
        print(f"EURAUD Signal: {signal} ({confidence:.1%})")
        print(f"Is Hidden in DB: {is_hidden} (Expected: 1)")
        print(f"Outcome: {outcome} (Expected: ACTIVE)")
        
        # Verify bridge cannot see it
        recent = db.get_active_signals(symbol="EURAUD", include_hidden=False)
        found_in_ui = any(s['id'] == sig_id for s in recent)
        print(f"Visible in UI/Bridge: {found_in_ui} (Expected: False)")
        
        # Verify it IS in the DB if we look for hidden
        all_active = db.get_active_signals(symbol="EURAUD", include_hidden=True)
        found_in_db = any(s['id'] == sig_id for s in all_active)
        print(f"Found in Hidden DB: {found_in_db} (Expected: True)")

    print("\n--- TEST 2: Certified Live Pair (GOLD) ---")
    # GOLD is APPROVED.
    res_gold = engine.predict_symbol("GOLD", save_to_db=True)
    if res_gold:
        sig_id = res_gold.get('id')
        is_hidden = res_gold.get('is_hidden')
        print(f"GOLD Signal: {res_gold['signal']} ({res_gold['confidence']:.1%})")
        print(f"Is Hidden in DB: {is_hidden} (Expected: 0)")
        
        recent_gold = db.get_active_signals(symbol="GOLD", include_hidden=False)
        found_in_ui = any(s['id'] == sig_id for s in recent_gold)
        print(f"Visible in UI/Bridge: {found_in_ui} (Expected: True)")

if __name__ == "__main__":
    test_shadow_certification()
