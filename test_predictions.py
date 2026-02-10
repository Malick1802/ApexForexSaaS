"""Test actual model predictions to see what's happening."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.inference import InferenceEngine
import logging

logging.basicConfig(level=logging.INFO)

print("Testing model predictions...")
print("="*70)

engine = InferenceEngine(confidence_threshold=0.50)

# Test EURUSD manually
test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

for symbol in test_symbols:
    print(f"\n{symbol}:")
    result = engine.predict_symbol(symbol, save_to_db=False)
    
    if result:
        print(f"  Signal: {result['signal']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  BUY prob: {result.get('buy_prob', 0):.1%}")
        print(f"  SELL prob: {result.get('sell_prob', 0):.1%}")
    else:
        print("  No high-confidence signal detected")

print("\n" + "="*70)
