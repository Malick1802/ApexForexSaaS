
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.inference import InferenceEngine
from data_pipeline.engine import DataEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Diagnose")

def diagnose(symbol="EURUSD"):
    print(f"\n{'='*50}")
    print(f"DIAGNOSING: {symbol}")
    print(f"{'='*50}")
    
    engine = InferenceEngine()
    
    # 1. Test Data Fetch
    print("\n1. FETCHING DATA...")
    days = 14
    df = engine.data_engine.fetch(symbol, interval="1h", days=days)
    if df is not None:
        print(f"SUCCESS: Fetched {len(df)} rows")
        print(f"Start: {df.index[0]}")
        print(f"End:   {df.index[-1]}")
    else:
        print("FAILED: No data fetched")
        return

    # 2. Test Feature Extraction
    print("\n2. EXTRACTING FEATURES...")
    base_features = engine.feature_engineer.extract_features(df)
    print(f"Extracted {len(base_features.columns)} base features")
    
    # 3. Test Prediction
    print("\n3. RUNNING PREDICTION...")
    # This calls the full predict_symbol including padding and sequences
    result = engine.predict_symbol(symbol, save_to_db=False, win_rate="70%", allow_stale=True)
    
    if result:
        print(f"\nSUCCESS: Prediction generated!")
        print(f"Signal:     {result['signal']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Trades:     {result['model_trades']}")
        print(f"Rows after dropna: (manual check needed in logs or by modifying code)")
    else:
        print("\nFAILED: predict_symbol returned None")
        print("Possible reasons: len(df) < 60, stale data, or crash in sequences")

if __name__ == "__main__":
    diagnose("EURUSD")
    diagnose("AUDCAD")
