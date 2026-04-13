import sys
import yaml
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core.inference import InferenceEngine
from data_pipeline.engine import DataEngine

def debug_ui_volume():
    symbol = "EURUSD"
    win_rate = "90%"
    
    # Initialize Engine
    print("--- 1. Initializing InferenceEngine ---")
    inf_engine = InferenceEngine()
    
    # Mimic APP.PY call
    print(f"--- 2. Calling predict_symbol({symbol}, win_rate={win_rate}) ---")
    result = inf_engine.predict_symbol(
        symbol, 
        save_to_db=False,
        win_rate=win_rate, 
        allow_stale=True
    )
    
    if result:
        print("\n--- 3. Result Dictionary (Subset) ---")
        print(f"Signal: {result.get('signal')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Winning Tier: {result.get('winning_tier')}")
        print(f"Model Trades (Raw Key): {result.get('model_trades', 'KEY_MISSING')}")
        
        # Check specifically for 0 vs None
        mt = result.get('model_trades')
        print(f"Type of model_trades: {type(mt)}")
        print(f"Value of model_trades: {mt}")
    else:
        print("\n[FAIL] predict_symbol returned None")

if __name__ == "__main__":
    try:
        debug_ui_volume()
    except Exception as e:
        print(f"CRASH: {e}")
