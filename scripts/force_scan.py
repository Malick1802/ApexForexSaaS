
import sys
import os
import logging
import asyncio

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.inference import InferenceEngine
from core.database import SignalDatabase
from data_pipeline import DataEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def force_scan():
    print("--- FORCING MARKET SCAN ---")
    
    # Initialize
    db = SignalDatabase()
    engine = DataEngine()
    # Load with default threshold to catch everything
    inf_engine = InferenceEngine(model_dir="models/trained", confidence_threshold=0.50) 
    
    pairs = engine.get_all_pairs()
    print(f"Scanning {len(pairs)} pairs...")
    
    for symbol in pairs:
        try:
            print(f"Scanning {symbol}...")
            # Use '90%' to match Dashboard/Executive default
            result = inf_engine.predict_symbol(symbol, save_to_db=True, win_rate="90%")
            
            if result:
                print(f"  > {symbol}: {result['signal']} ({result['confidence']:.2%})")
            else:
                print(f"  > {symbol}: No Result")
                
        except Exception as e:
            print(f"  > {symbol}: Error {e}")
            
    print("--- SCAN COMPLETE ---")

if __name__ == "__main__":
    force_scan()
