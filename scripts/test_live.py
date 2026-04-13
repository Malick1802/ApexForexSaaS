import sys
import logging
sys.path.insert(0, '.')
from core.inference import InferenceEngine

# Disable extra logging for clean output
logging.getLogger("data_pipeline.engine").setLevel(logging.WARNING)
logging.getLogger("models.foundation_trainer").setLevel(logging.WARNING)

print('Testing Live Inference with Phase 4 Regime Upgrades...')
print('-'*70)
engine = InferenceEngine()

for sym in ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDCAD']:
    print(f'\nScanning {sym}...')
    res = engine.predict_symbol(sym, save_to_db=False)
    if res:
        print(f"  Signal: {res['signal']}")
        print(f"  Regime: {res.get('regime', 'N/A')}")
        print(f"  Regime Threshold: {res.get('regime_threshold', 0):.0%}")
        print(f"  Model Confidence: {res['confidence']:.1%}")
        print(f"  Output Probs -> BUY: {res.get('buy_prob', 0):.1%} | SELL: {res.get('sell_prob', 0):.1%} | WAIT: {res.get('wait_prob', 0):.1%}")
    else:
        print('  Result: Blocked by Regime or No high-confidence signal')
print('\n'+'-'*70)
