import shutil
import os
from pathlib import Path

# List of pairs to reset (Alphabetical from EURJPY onwards)
pairs_to_reset = [
    "EURJPY", "EURNZD", "EURSGD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDHKD", "USDJPY", "USDSGD"
]

base_dir = Path("models")
specialist_dir = base_dir / "specialist"

print(f"Reseting {len(pairs_to_reset)} pairs...")

for pair in pairs_to_reset:
    # 1. Delete Expert Models (90, 95)
    pair_path = base_dir / pair
    if pair_path.exists():
        for target in ["90", "95"]:
            target_path = pair_path / target
            if target_path.exists():
                print(f"Deleting {target_path}")
                shutil.rmtree(target_path)
    
    # 2. Delete Specialist Models (Base) - Per "Same Technique" request
    # This forces a re-evaluation of the base model quality (60% check)
    sp_path = specialist_dir / pair
    if sp_path.exists():
        print(f"Deleting {sp_path}")
        shutil.rmtree(sp_path)

print("Reset complete. Ready for retraining.")
