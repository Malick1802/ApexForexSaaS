"""Quick test of inference engine."""
from core.inference import InferenceEngine
import logging

logging.basicConfig(level=logging.INFO)

# Test inference
engine = InferenceEngine(model_dir='models/binary', confidence_threshold=0.60)
result = engine.predict_symbol('EURUSD', save_to_db=True)

print('\n' + '='*60)
print('SIGNAL GENERATION TEST')
print('='*60)

if result:
    print(f"✓ Symbol: {result['symbol']}")
    print(f"✓ Signal: {result['signal']}")
    print(f"✓ Confidence: {result['confidence']:.1%}")
    print(f"✓ Entry Price: {result['price_at_signal']:.5f}")
    print(f"✓ Take Profit: {result['tp_price']:.5f} (+{result['tp_pips']} pips)")
    print(f"✓ Stop Loss: {result['sl_price']:.5f} (-{result['sl_pips']} pips)")
    print(f"✓ Saved to database: Yes")
else:
    print("ℹ No signal detected (WAIT or low confidence)")

print('='*60)
