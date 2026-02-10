import os
import json
import pandas as pd

MODEL_DIR = "models/specialist"

results = []

if not os.path.exists(MODEL_DIR):
    print(f"Directory {MODEL_DIR} does not exist.")
    exit()

for symbol in os.listdir(MODEL_DIR):
    metrics_path = os.path.join(MODEL_DIR, symbol, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                
            # Extract key metrics
            # Some metrics might be under 'folds' or averaged
            # Assuming the JSON has top-level average or we average the folds
            
            # If it's the structure from EnhancedPairTrainer, it might vary.
            # Let's check typical keys.
            
            win_rate = data.get('win_rate', 0.0)
            filtered_rate = data.get('filtered_win_rate', 0.0)
            
            # If 0, maybe it's a list of folds?
            if win_rate == 0 and isinstance(data, list):
                # Valid logic if data is list of fold metrics
                df_folds = pd.DataFrame(data)
                win_rate = df_folds['win_rate'].mean()
                filtered_rate = df_folds['filtered_win_rate'].mean()
            elif isinstance(data, dict) and 'mean_metrics' in data:
                 win_rate = data['mean_metrics'].get('win_rate', 0)
                 filtered_rate = data['mean_metrics'].get('filtered_win_rate', 0)

            results.append({
                "Symbol": symbol,
                "Win Rate": f"{win_rate:.1%}",
                "Filtered Win Rate": f"{filtered_rate:.1%}"
            })
        except Exception as e:
            print(f"Error reading {symbol}: {e}")

if results:
    df = pd.DataFrame(results)
    # Sort by Filtered Win Rate descending
    df = df.sort_values("Filtered Win Rate", ascending=False)
    print(df.to_markdown(index=False))
else:
    print("No trained models found yet.")
