import os
import sys
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core.inference import InferenceEngine

engine = InferenceEngine(confidence_threshold=0.8)
pairs = ['EURUSD', 'GBPUSD', 'AUDJPY', 'USDJPY', 'USDCHF']
print("--- 🛰️ Current Market Conviction Analysis ---")
for p in pairs:
    try:
        res = engine.predict_symbol(p, save_to_db=False)
        print(f"{p}: {res['confidence']:.1%} ({res['signal']})")
    except Exception as e:
        print(f"{p}: Error ({e})")
