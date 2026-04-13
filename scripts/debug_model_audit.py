
import os
import json

MODELS_DIR = "models"

def debug_audit():
    print(f"Scanning {MODELS_DIR} for config.json...")
    count = 0
    
    for root, dirs, files in os.walk(MODELS_DIR):
        if "config.json" in files:
            path = os.path.join(root, "config.json")
            try:
                # Check for "90" in path to filter for new models
                if os.sep + "90" + os.sep in path or "90" in root.split(os.sep):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    sym = data.get('symbol', 'Unknown')
                    typ = data.get('type', 'Unknown')
                    wr = data.get('win_rate', 0)
                    trs = data.get('trades', 0)
                    
                    print(f"FOUND: {sym} {typ} | WR: {wr:.2%} | Vol: {trs} | Path: {root}")
                    count += 1
            except Exception as e:
                print(f"Error reading {path}: {e}")

    print(f"Total Configs Found: {count}")

if __name__ == "__main__":
    debug_audit()
