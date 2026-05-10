import os
import sys
sys.path.insert(0, os.getcwd())
from core.performance_gate import get_performance_gate
import json

# Force recompute
gate = get_performance_gate()
print("Recomputing Performance Matrix (14-day window)...")
gate.recompute_from_db(lookback_days=14)

# Verify EURCAD status
status = gate.get_tier_status("EURCAD", "BUY", 1.0)
print(f"EURCAD BUY (100% Tier) Status: {status}")

# Check whitelist file content
with open("config/trading_whitelist.json", "r") as f:
    data = json.load(f)
    eurcad_buy_100 = data['performance_matrix'].get('EURCAD', {}).get('BUY', {}).get('100', {})
    print(f"EURCAD BUY 100% Data: {json.dumps(eurcad_buy_100, indent=2)}")
