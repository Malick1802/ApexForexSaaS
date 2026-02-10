"""Test loading specialist models."""
from core.inference import InferenceEngine
import logging

logging.basicConfig(level=logging.INFO)

engine = InferenceEngine(confidence_threshold=0.60)

# Test a few specialist models
test_symbols = ['AUDCAD', 'EURGBP', 'GBPJPY', 'NZDUSD']

print('\n' + '='*60)
print('SPECIALIST MODEL DETECTION TEST')
print('='*60)

loaded_count = 0
for symbol in test_symbols:
    models = engine.load_enhanced_model(symbol)
    if models:
        print(f"✓ {symbol}: Model loaded successfully")
        loaded_count += 1
    else:
        print(f"✗ {symbol}: Model not found")

print('='*60)
print(f'{loaded_count}/{len(test_symbols)} models detected')
print('='*60)
