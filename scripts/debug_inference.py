
import sys
import os
import logging
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.inference import InferenceEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_symbol(symbol="AUDUSD"):
    print(f"\n--- DEBUGGING {symbol} ---")
    
    # 1. Check User Default (90%)
    print("\n🔍 Checking '90%' Target (Dashboard Default)...")
    try:
        engine_90 = InferenceEngine(model_dir="models/trained", confidence_threshold=0.70)
        res_90 = engine_90.predict_symbol(symbol, win_rate="90%")
        if res_90:
            print(f"  RESULT: {res_90['signal']} (Conf: {res_90['confidence']:.1%})")
            print(f"  Model: {res_90.get('model_version', 'Unknown')}")
        else:
            print("  RESULT: No Signal (None)")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Check Sentinel Default (Apex)
    print("\n🔍 Checking 'Apex' Target (Sentinel Default)...")
    try:
        engine_apex = InferenceEngine(model_dir="models/trained", confidence_threshold=0.80)
        
        # Debug Data Age
        print("  Checking Data Freshness...")
        df = engine_apex.data_engine.fetch(symbol, interval="1h", days=5)
        if df is not None and not df.empty:
            last_time = df.index[-1]
            print(f"  Last Candle: {last_time}")
            is_stale = engine_apex._is_data_stale(last_time)
            print(f"  Is Stale? {is_stale}")
        else:
            print("  Data Fetch Failed or Empty!")

        res_apex = engine_apex.predict_symbol(symbol, win_rate="Apex")
        if res_apex:
            print(f"  RESULT: {res_apex['signal']} (Conf: {res_apex['confidence']:.1%})")
            print(f"  Model: {res_apex.get('model_version', 'Unknown')}")
        else:
            print("  RESULT: No Signal (None returned)")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n--- DONE ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_symbol(sys.argv[1])
    else:
        debug_symbol()
