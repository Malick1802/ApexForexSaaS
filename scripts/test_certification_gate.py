import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.core.inference import InferenceEngine

def test_certification_gate():
    engine = InferenceEngine(model_dir="models/expert")
    
    print("\n--- TEST: Certified Pair (GOLD) ---")
    # GOLD is approved in the performance_matrix at 60, 70, 80 tiers.
    res_gold = engine.predict_symbol("GOLD", save_to_db=False)
    if res_gold:
        is_proven = res_gold.get('is_proven')
        outcome = res_gold.get('outcome')
        print(f"GOLD Signal: {res_gold['signal']} ({res_gold['confidence']:.1%})")
        print(f"Is Proven: {is_proven} (Expected: 1)")
        print(f"Outcome: {outcome} (Expected: ACTIVE)")
    
    print("\n--- TEST: Uncertified Pair (NZDJPY) ---")
    # NZDJPY is currently BENCHED (not yet approved)
    res_nz = engine.predict_symbol("NZDJPY", save_to_db=False)
    if res_nz:
        is_proven = res_nz.get('is_proven')
        outcome = res_nz.get('outcome')
        print(f"NZDJPY Signal: {res_nz['signal']} ({res_nz['confidence']:.1%})")
        print(f"Is Proven: {is_proven} (Expected: 0)")
        print(f"Outcome: {outcome} (Expected: N/A)")

if __name__ == "__main__":
    test_certification_gate()
