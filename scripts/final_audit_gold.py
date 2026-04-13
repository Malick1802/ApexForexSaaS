import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from core.core.inference import InferenceEngine

engine = InferenceEngine()
res = engine.predict_symbol('GOLD', save_to_db=True, allow_stale=True)
print(f"--- 🛰️ FINAL GOLD AUDIT ---")
print(f"SIGNAL: {res['signal']}")
print(f"CONFIDENCE: {res['confidence']*100:.1f}%")
print(f"RAW CONFIDENCE: {res.get('raw_confidence', 0)*100:.1f}%")
print(f"REGIME: {res.get('regime')}")
print(f"TP/SL: {res.get('tp_pips')} / {res.get('sl_pips')} pips")
