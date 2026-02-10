import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_levels():
    engine = InferenceEngine(config_path="config.yaml")
    
    test_pairs = ["EURUSD", "GBPJPY", "AUDCHF"]
    
    print("\n" + "="*50)
    print("DYNAMIC SL/TP VERIFICATION")
    print("="*50)
    print(f"{'Pair':<10} | {'Entry':<10} | {'SL (Pips)':<10} | {'TP (Pips)':<10} | {'RR':<5}")
    print("-" * 50)
    
    for symbol in test_pairs:
        try:
            # Force a prediction (allow_stale=True for testing)
            # We use a low win_rate to ensure we get a "Result" object even if it's WAIT
            result = engine.predict_symbol(symbol, win_rate="60%", allow_stale=True)
            
            if result:
                # We can recalculate levels manually to see the comparison or just use what was returned
                # Note: predict_symbol now calculates these dynamically inside
                sl = result['sl_pips']
                tp = result['tp_pips']
                entry = result['price_at_signal']
                rr = tp / sl if sl != 0 else 0
                
                print(f"{symbol:<10} | {entry:<10.5f} | {sl:<10} | {tp:<10} | {rr:<5.1f}")
            else:
                print(f"{symbol:<10} | Failed to get result")
        except Exception as e:
            print(f"{symbol:<10} | Error: {e}")

    print("="*50 + "\n")

if __name__ == "__main__":
    verify_levels()
