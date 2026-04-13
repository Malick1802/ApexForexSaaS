
import os
import json
import glob
import pandas as pd

MODELS_DIR = "models"
TARGET_WIN_RATE = "90"

def audit_models():
    print(f"Auditing models in {MODELS_DIR}/*/ {TARGET_WIN_RATE} ...")
    
    results = []
    
    # Pattern to find all buy/sell configs
    # models/AUDCAD/90/BUY/config.json
    pattern = os.path.join(MODELS_DIR, "*", TARGET_WIN_RATE, "*", "config.json")
    files = glob.glob(pattern)
    
    if not files:
        print("No model configs found.")
        return

    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                
            symbol = data.get('symbol', 'Unknown')
            direction = data.get('type', 'Unknown')
            win_rate = data.get('win_rate', 0.0)
            trades = data.get('trades', 0)
            
            # Extract parent folder name if symbol missing
            if symbol == 'Unknown':
                parts = f.split(os.sep)
                symbol = parts[-4] # models/SYMBOL/90/BUY/config.json
            
            results.append({
                'Symbol': symbol,
                'Direction': direction,
                'Win Rate': win_rate,
                'Trades': trades
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not results:
        print("No valid data found.")
        return

    df = pd.DataFrame(results)
    
    # Sort by Win Rate DESC
    df = df.sort_values(by='Win Rate', ascending=False)
    
    print("-" * 60)
    print(f" NEW MODELS REPORT (Feb 10) - Target {TARGET_WIN_RATE}%")
    print("-" * 60)
    print(f"Total Models: {len(df)}")
    print(f"Avg Win Rate: {df['Win Rate'].mean():.1%}")
    print(f"Total Trades: {df['Trades'].sum()}")
    print("-" * 60)
    
    # Print top 20
    print(df.to_string(index=False, formatters={
        'Win Rate': '{:.1%}'.format
    }))
    print("-" * 60)

if __name__ == "__main__":
    audit_models()
