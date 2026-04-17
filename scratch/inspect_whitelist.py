"""Inspect the trading whitelist structure and fix CrudeOIL key mismatch."""
import json
from pathlib import Path

wl_path = Path("config/trading_whitelist.json")
if not wl_path.exists():
    print("Whitelist not found!")
    exit()

data = json.loads(wl_path.read_text())
matrix = data.get("performance_matrix", {})

print("Top-level keys:", list(data.keys()))
print(f"\nTotal pairs in performance_matrix: {len(matrix)}")
print("\nAll pair keys in matrix:")
for k in sorted(matrix.keys()):
    print(f"  '{k}'")
