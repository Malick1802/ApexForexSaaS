import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHITELIST_PATH = PROJECT_ROOT / "config" / "trading_whitelist.json"

def migrate():
    if not WHITELIST_PATH.exists():
        print("No whitelist found to migrate.")
        return

    with open(WHITELIST_PATH, "r") as f:
        data = json.load(f)

    matrix = data.get("performance_matrix", {})
    new_matrix = {}

    for symbol, content in matrix.items():
        new_matrix[symbol] = {"BUY": {}, "SELL": {}, "ALL": {}}
        
        # Move existing BUY/SELL if they exist
        if "BUY" in content:
            new_matrix[symbol]["BUY"] = content["BUY"]
        if "SELL" in content:
            new_matrix[symbol]["SELL"] = content["SELL"]
        if "ALL" in content:
            new_matrix[symbol]["ALL"] = content["ALL"]

        # Move legacy top-level tiers (60, 70, 80, 90, 100) to ALL
        for tier in ["60", "70", "80", "90", "100"]:
            if tier in content:
                new_matrix[symbol]["ALL"][tier] = content[tier]
                print(f"Migrated {symbol} legacy tier {tier} to ALL.")

    data["performance_matrix"] = new_matrix
    
    with open(WHITELIST_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    print("\nMigration Complete. Whitelist normalized.")

if __name__ == "__main__":
    migrate()
