import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core.inference import InferenceEngine

def test_top():
    print("--- LIVE MARKET CONVICTION SNAPSHOT ---")
    engine = InferenceEngine(confidence_threshold=0.50) # Low threshold just to force output
    
    test_pairs = ["EURUSD", "GBPUSD", "USDJPY", "GOLD", "AUDUSD"]
    
    for pair in test_pairs:
        try:
            signal = engine.predict_symbol(pair)
            if signal:
                print(f"{pair}: {signal['direction']} @ {signal['confidence']:.1%} confidence")
            else:
                print(f"{pair}: No clear direction or model not certified.")
        except Exception as e:
            print(f"{pair}: Error checking model - {e}")
            
if __name__ == "__main__":
    test_top()
