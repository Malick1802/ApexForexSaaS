"""Diagnose exactly why CrudeOIL signals stay shadow."""
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')

# 1. Show the whitelist entry
wl_path = Path("config/trading_whitelist.json")
data = json.loads(wl_path.read_text())
matrix = data["performance_matrix"]
crude_entry = matrix.get("CrudeOIL", {})
print("CrudeOIL whitelist entry:")
print(json.dumps(crude_entry, indent=2))

# 2. Test the performance gate directly
print("\nPerformance gate test:")
try:
    from core.performance_gate import get_performance_gate
    gate = get_performance_gate()
    for conf in [0.60, 0.65, 0.67, 0.70, 0.80]:
        result = gate.is_tier_approved("CrudeOIL", "BUY", conf)
        print(f"  is_tier_approved('CrudeOIL', 'BUY', {conf:.2f}) => {result}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Check what get_tier_status returns
print("\nTier status:")
try:
    from core.performance_gate import get_performance_gate
    gate = get_performance_gate()
    status = gate.get_tier_status("CrudeOIL", "BUY", 0.67)
    print(f"  get_tier_status => {status}")
except Exception as e:
    print(f"  Error: {e}")
