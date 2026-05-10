import os
import sys
import json
sys.path.insert(0, os.getcwd())
from core.performance_gate import get_performance_gate

gate = get_performance_gate()
print("EURCAD BUY Tiers Status:")
for t in [50, 55, 60, 70, 80, 90, 100]:
    status = gate.get_tier_status("EURCAD", "BUY", t)
    print(f"  Tier {t}%: {status}")

with open("config/trading_whitelist.json", "r") as f:
    data = json.load(f)
    eurcad_buy = data['performance_matrix'].get('EURCAD', {}).get('BUY', {})
    print(f"\nEURCAD BUY Data:\n{json.dumps(eurcad_buy, indent=2)}")
