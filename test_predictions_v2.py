"""Test predictions with internal log capture."""
import sys
import os
from pathlib import Path

# Redirect stdout/stderr to file
sys.stdout = open('internal_log.txt', 'w')
sys.stderr = sys.stdout

print("STARTING TEST")

sys.path.insert(0, str(Path(__file__).parent))
from core.inference import InferenceEngine
import logging

# output to stdout (which is now file)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

print("Testing model predictions...")
print("="*70)

engine = InferenceEngine(confidence_threshold=0.50)

test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

for symbol in test_symbols:
    print(f"\n{symbol}:")
    try:
        result = engine.predict_symbol(symbol, save_to_db=False)
        
        if result:
            print(f"  Signal: {result['signal']}")
            print(f"  Confidence: {result['confidence']:.1%}")
        else:
            print("  No high-confidence signal detected")
    except Exception as e:
        print(f"CRASH: {e}")

print("\n" + "="*70)
print("DONE")
