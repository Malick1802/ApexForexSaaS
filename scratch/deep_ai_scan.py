import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference import InferenceEngine
import json

def test_ai():
    print("--- APEX AI DEEP SCAN ---")
    eng = InferenceEngine()
    
    # Check top 3 pairs
    symbols = ['EURUSD', 'GBPUSD', 'GOLD', 'XAUUSD']
    for sym in symbols:
        try:
            # predict_symbol returns a single Dict or None
            res = eng.predict_symbol(sym, use_cache=False)
            if res:
                print(f"{sym}:")
                print(f"  Signal:     {res['signal']}")
                print(f"  Confidence: {res['confidence']:.2%}")
                print(f"  Regime:     {res.get('regime', 'UNKNOWN')}")
                print(f"  Hurdle:     {res.get('regime_threshold', 0.60):.0%}")
                print(f"  Probabilities: Buy={res.get('buy_prob', 0):.1%}, Sell={res.get('sell_prob', 0):.1%}, Wait={res.get('wait_prob', 0):.1%}")
                print(f"  Outcome Status: {res.get('outcome', 'N/A')}")
            else:
                print(f"{sym}: No data or stale data (Market might be quiet)")
        except Exception as e:
            print(f"{sym}: Error: {e}")

if __name__ == "__main__":
    test_ai()
