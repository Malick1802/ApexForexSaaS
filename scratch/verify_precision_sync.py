import json
import pandas as pd
from pathlib import Path

def verify_precision_sync():
    WHITELIST_PATH = Path("config/trading_whitelist.json")
    
    if not WHITELIST_PATH.exists():
        print("Error: Whitelist JSON not found.")
        return

    with open(WHITELIST_PATH, "r") as f:
        data = json.load(f)
        matrix = data.get("performance_matrix", {})
        
    print(f"--- PRECISION MATRIX AUDIT ({data.get('last_updated', 'Unknown')}) ---")
    
    # Check CrudeOIL and NZDJPY specifically
    for sym in ['CrudeOIL', 'NZDJPY', 'NZDUSD']:
        if sym in matrix:
            print(f"\n[{sym}]")
            for key, val in matrix[sym].items():
                if isinstance(val, dict) and any(t in key for t in ['60', '70', '80', '90', '100', 'BUY', 'SELL']):
                    if key in ['BUY', 'SELL']:
                        print(f"  {key}:")
                        for t, stats in val.items():
                            if isinstance(stats, dict):
                                print(f"    Tier {t}%: {stats.get('trades', 0)} Trades | Win Rate: {stats.get('accuracy', 0):.1%} | Status: {stats.get('status')}")
                    else:
                        print(f"  Tier {key}%: {val.get('trades', 0)} Trades | Win Rate: {val.get('accuracy', 0):.1%} | Status: {val.get('status')}")
        else:
            print(f"\n[{sym}]: No data in matrix.")

if __name__ == "__main__":
    verify_precision_sync()
