
import os
import json
import pandas as pd

MODELS_DIR = "models"
OUTPUT_FILE = "model_audit_report.md"

def generate_report():
    print(f"Scanning {MODELS_DIR}...")
    results = []
    
    for root, dirs, files in os.walk(MODELS_DIR):
        if "config.json" in files:
            path = os.path.join(root, "config.json")
            try:
                if os.sep + "90" + os.sep in path or "90" in root.split(os.sep):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    results.append({
                        'Symbol': data.get('symbol', 'Unknown'),
                        'Direction': data.get('type', 'Unknown'),
                        'Win_Rate': data.get('win_rate', 0.0),
                        'Trades': data.get('trades', 0)
                    })
            except: pass

    if not results:
        print("No data.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by='Win_Rate', ascending=False)
    
    avg_wr = df['Win_Rate'].mean()
    total_trades = df['Trades'].sum()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# 📊 New Model Performance Report (Feb 10, 2026)\n\n")
        f.write(f"**Total Models:** {len(df)}\n")
        f.write(f"**Average Win Rate:** {avg_wr:.2%}\n")
        f.write(f"**Total Trade Volume:** {total_trades}\n\n")
        
        f.write("| Symbol | Direction | Win Rate | Trades |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for _, row in df.iterrows():
            f.write(f"| **{row['Symbol']}** | {row['Direction']} | {row['Win_Rate']:.2%} | {row['Trades']} |\n")
            
    print(f"Report written to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
