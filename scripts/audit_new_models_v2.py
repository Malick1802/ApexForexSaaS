
import os
import json
import pandas as pd

MODELS_DIR = "models"
TARGET_WIN_RATE = 0.90  # Look for configs with target_win_rate >= 90

def audit_models_robust():
    print(f"Walking {MODELS_DIR} to find new models...")
    
    results = []
    
    for root, dirs, files in os.walk(MODELS_DIR):
        if "config.json" in files:
            full_path = os.path.join(root, "config.json")
            try:
                # Check if this is a "90" model folder
                # heuristics: path contains "90"
                if "90" not in root.split(os.sep):
                    continue

                with open(full_path, 'r') as f:
                    data = json.load(f)
                
                # Check creation date if available (optional)
                # created = data.get('created_at', '')
                
                results.append({
                    'Symbol': data.get('symbol', 'Unknown'),
                    'Direction': data.get('type', 'Unknown'),
                    'Win Rate': data.get('win_rate', 0.0),
                    'Trades': data.get('trades', 0),
                    'Path': root
                })
            except Exception as e:
                # print(f"Error reading {full_path}: {e}")
                pass

    if not results:
        print("No models found matching criteria.")
        return

    df = pd.DataFrame(results)
    
    # Filter for high win rates if needed
    # df = df[df['Win Rate'] >= 0.90]
    
    df = df.sort_values(by='Win Rate', ascending=False)
    
    print("-" * 60)
    print(f" NEW MODELS REPORT (Detected {len(df)} models)")
    print("-" * 60)
    print(f"Avg Win Rate: {df['Win Rate'].mean():.1%}")
    print(f"Total Trades: {df['Trades'].sum()}")
    print("-" * 60)
    
    print(df[['Symbol', 'Direction', 'Win Rate', 'Trades']].to_string(index=False, formatters={
        'Win Rate': '{:.1%}'.format
    }))
    print("-" * 60)

if __name__ == "__main__":
    audit_models_robust()
