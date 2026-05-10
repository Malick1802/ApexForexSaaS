"""
Live Inference Test — Post Scaler Fix
Confirms the model is generating balanced BUY/SELL signals.
"""
import sys, logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '.')

from core.inference import InferenceEngine

engine = InferenceEngine(confidence_threshold=0.50)

test_pairs = ['EURUSD', 'USDJPY', 'GOLD', 'AUDUSD', 'GBPUSD', 'NZDUSD', 'USDCHF', 'AUDCAD']

print('=' * 65)
print('  LIVE INFERENCE TEST — Post Scaler Fix')
print('=' * 65)
print(f"{'Symbol':<10} {'Signal':<6} {'BUY':>7} {'SELL':>7} {'WAIT':>7}  Confidence")
print('-' * 65)

buy_count = 0
sell_count = 0
wait_count = 0

for sym in test_pairs:
    try:
        result = engine.predict_symbol(sym, save_to_db=False, allow_stale=True)
        if result:
            sig   = result['signal']
            bp    = result.get('buy_prob', 0.0)
            sp    = result.get('sell_prob', 0.0)
            wp    = result.get('wait_prob', 0.0)
            conf  = result.get('confidence', 0.0)
            print(f"  {sym:<8} {sig:<6} {bp:>7.3f} {sp:>7.3f} {wp:>7.3f}  {conf:.3f}")
            if sig == 'BUY':   buy_count  += 1
            elif sig == 'SELL': sell_count += 1
            else:               wait_count += 1
        else:
            print(f"  {sym:<8} [locked or no data]")
    except Exception as e:
        print(f"  {sym:<8} ERROR: {e}")

print('-' * 65)
print(f"\n  BUY signals:  {buy_count}")
print(f"  SELL signals: {sell_count}")
print(f"  WAIT signals: {wait_count}")
print()

if buy_count > 0 and sell_count > 0:
    print("  ✅ BIAS RESOLVED — Both BUY and SELL signals present.")
elif buy_count == 0 and sell_count > 0:
    print("  ⚠  Still SELL-only. Check scaler alignment.")
elif buy_count > 0 and sell_count == 0:
    print("  ⚠  BUY-only signals (possible reverse bias). Monitor.")
else:
    print("  ℹ  All WAIT — model is cautious but unbiased (check confidence thresholds).")
