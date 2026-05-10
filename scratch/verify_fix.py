import os
import sys
sys.path.insert(0, os.getcwd())
from core.performance_gate import get_performance_gate

gate = get_performance_gate()
symbol = "EURCAD"
direction = "BUY"
confidence = 1.0

is_approved = gate.is_tier_approved(symbol, direction, confidence)
print(f"EURCAD BUY 100% Confidence Approved: {is_approved}")

# Also check 70%
is_approved_70 = gate.is_tier_approved(symbol, direction, 0.7)
print(f"EURCAD BUY 70% Confidence Approved: {is_approved_70}")
