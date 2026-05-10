import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
sys.path.insert(0, os.getcwd())
from core.inference import InferenceEngine

# Setup mock engine
engine = InferenceEngine()

# Mock a pair that is proven but has LOW raw conviction (45%)
# We'll simulate the internal state of predict_symbol by running a snippet
symbol = "EURCAD"
raw_confidence = 0.45
final_confidence = 0.90 # High Win Rate substitute
buy_threshold = 0.70
is_tier_proven = True

has_raw_edge = (raw_confidence >= 0.60)
is_authorized = False
if (has_raw_edge and (final_confidence >= buy_threshold or is_tier_proven)):
    is_authorized = True

print(f"--- Low Conviction Test (Raw: {raw_confidence:.1%}, WinRate: {final_confidence:.1%}) ---")
print(f"Is Authorized: {is_authorized}")
print(f"Expected: False")

# Mock a pair that is proven and has HIGH raw conviction (65%)
raw_high = 0.65
has_raw_high = (raw_high >= 0.60)
is_authorized_high = False
if (has_raw_high and (final_confidence >= buy_threshold or is_tier_proven)):
    is_authorized_high = True

print(f"\n--- High Conviction Test (Raw: {raw_high:.1%}, WinRate: {final_confidence:.1%}) ---")
print(f"Is Authorized: {is_authorized_high}")
print(f"Expected: True")

# Check Tier Rounding
actual_tier_low = int(raw_confidence * 10) * 10
print(f"\nTier for {raw_confidence:.1%}: {actual_tier_low}% (Expected: 40%)")
