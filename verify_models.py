"""Verify high-performance binary models are loaded."""
from core.inference import InferenceEngine
import logging

logging.basicConfig(level=logging.INFO)

engine = InferenceEngine(confidence_threshold=0.60)

# Test loading models
test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

print('\n' + '='*70)
print('HIGH-PERFORMANCE MODEL VERIFICATION')
print('='*70)

for symbol in test_symbols:
    models = engine.load_binary_models(symbol)
    if models:
        print(f"✓ {symbol}: Binary models loaded from {models.get('loaded_from', 'specialist')}")
    else:
        print(f"✗ {symbol}: Failed to load")

print('='*70)
print('Expected: All models loaded from models/specialist')
print('='*70)
