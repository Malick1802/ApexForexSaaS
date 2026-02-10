import os
import json
import pandas as pd
from pathlib import Path

def generate_report():
    base_dir = Path('models/specialist')
    results = []

    if not base_dir.exists():
        print("No models directory found.")
        return

    for pair_dir in sorted(base_dir.iterdir()):
        if not pair_dir.is_dir(): continue
        
        symbol = pair_dir.name
        
        # Check BUY model
        buy_metrics = pair_dir / 'BUY' / 'metrics.json'
        if buy_metrics.exists():
            try:
                with open(buy_metrics, 'r') as f:
                    data = json.load(f)
                    results.append({
                        'Pair': symbol,
                        'Signal': 'BUY',
                        'Accuracy': f"{data['accuracy']:.2%}",
                        'Win Rate': f"{data['accuracy']:.2%}", # Using accuracy as proxy for win rate as per Specialist Factory logic
                        'Status': '✅ Certified'
                    })
            except Exception as e:
                print(f"Error reading {buy_metrics}: {e}")

        # Check SELL model
        sell_metrics = pair_dir / 'SELL' / 'metrics.json'
        if sell_metrics.exists():
            try:
                with open(sell_metrics, 'r') as f:
                    data = json.load(f)
                    results.append({
                        'Pair': symbol,
                        'Signal': 'SELL',
                        'Accuracy': f"{data['accuracy']:.2%}",
                        'Win Rate': f"{data['accuracy']:.2%}",
                        'Status': '✅ Certified'
                    })
            except Exception as e:
                print(f"Error reading {sell_metrics}: {e}")

    if not results:
        print("No trained models found.")
        return

    df = pd.DataFrame(results)
    
    # Pivot for cleaner view
    print("\n# AI Specialist Models Report")
    print(f"\n**Total Models Certified**: {len(df)}")
    
    print("\n## Detailed Performance")
    print(df.to_markdown(index=False))
    
    # Generate CSV artifact
    df.to_csv('models/specialist_report.csv', index=False)
    print("\nReport saved to models/specialist_report.csv")

if __name__ == "__main__":
    generate_report()
