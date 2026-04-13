import time
import re
import os
import json
from pathlib import Path
from datetime import datetime

# Log file path
LOG_FILE = Path("logs/specialist_progressive.log")
STATUS_FILE = Path("fleet_status_v2.md")
MODELS_DIR = Path("models/specialist")

# Regex patterns
PAIR_PATTERN = re.compile(r"Starting Specialist Factory for (\w+) \((\w+)\)")
FOLD_PATTERN = re.compile(r"Fold (\d+): Acc=([\d\.]+)%, Loss=([\d\.]+), Trades=(\d+), WR=([\d\.]+)%")
CERTIFIED_PATTERN = re.compile(r"🏆 GOLDEN CERTIFIED")

# State tracking
# State tracking
active_worker = {"pair": "N/A", "dir": "-", "updated": datetime.min, "metric": "-"}
recent_activity = []

def get_certified_models():
    """Scan the filesystem for trained models (Source of Truth)."""
    certified = []
    if not MODELS_DIR.exists():
        return []
        
    for symbol_dir in MODELS_DIR.iterdir():
        if not symbol_dir.is_dir(): continue
        symbol = symbol_dir.name
        
        for signal_dir in symbol_dir.iterdir():
            if not signal_dir.is_dir(): continue
            signal = signal_dir.name
            
            config_path = signal_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                        wr = data.get('win_rate', 0)
                        trades = data.get('trades', 0)
                        acc = data.get('accuracy', 0)
                        timestamp = data.get('timestamp', 'N/A')
                        
                        certified.append({
                            'symbol': symbol,
                            'signal': signal,
                            'wr': wr,
                            'trades': trades,
                            'acc': acc,
                            'ts': timestamp
                        })
                except:
                    pass
    
    certified.sort(key=lambda x: x['wr'], reverse=True)
    return certified

def parse_log():
    global recent_activity
    if not LOG_FILE.exists():
        return
        
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()[-3000:] 
        
    activity_stream = []
    
    current_pair = "N/A"
    current_dir = "-"
    current_metric = "-"
    
    for line in lines:
        pair_match = PAIR_PATTERN.search(line)
        if pair_match:
            current_pair, current_dir = pair_match.groups()
            current_metric = "-"

        fold_match = FOLD_PATTERN.search(line)
        if fold_match:
            fold, acc, loss, trades, wr = fold_match.groups()
            current_metric = f"Fold {fold}: WR {float(wr):.1f}%, Acc {float(acc):.1f}%, Vol {trades}"
            
            activity_stream.append({
                'fold': fold,
                'acc': acc,
                'trades': trades,
                'wr': wr,
                'raw': line.strip()
            })
            
    recent_activity = activity_stream[-5:]
    
    # Store dynamic active worker
    active_worker["pair"] = current_pair
    active_worker["dir"] = current_dir
    active_worker["metric"] = current_metric
    active_worker["updated"] = datetime.now()

def update_artifact():
    certified = get_certified_models()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# 🚀 Specialist Training Fleet Status
**Updated:** {timestamp}
**Target:** Win Rate ≥ 70%, Accuracy ≥ 70%, Trades ≥ 600

---

## 🛠️ Active Workers (Dynamic Tracking)

| Worker | Current Pair | Direction | Latest Metric | Status |
|:---:|:---|:---|:---|:---|
| **W1** | **{active_worker['pair']}** | {active_worker['dir']} | {active_worker['metric']} | 🟡 Training |

---

## 📊 Recent Validation Activity (Stream)
*Live stream of validation results from the fleet:*

| Fold | Accuracy | Win Rate | Trades |
|:---:|:---:|:---:|:---:|
"""
    
    if recent_activity:
        for act in recent_activity:
            content += f"| Fold {act['fold']} | {float(act['acc']):.2f}% | {float(act['wr']):.2f}% | {act['trades']} |\n"
    else:
        content += "| - | - | - | - |\n"

    content += """
---

## 🏆 Certified Models (Golden Signals)
*Models passing strict 70% Win Rate + 70% Accuracy + 600 Trades criteria:*

| Pair | Signal | Win Rate | Accuracy | Trades |
|:---|:---|:---:|:---:|:---:|
"""

    if certified:
        for m in certified:
            wr_str = f"{m['wr']*100:.2f}%" if m['wr'] < 1 else f"{m['wr']:.2f}%"
            acc_str = f"{m['acc']*100:.2f}%" if m['acc'] < 1 else f"{m['acc']:.2f}%"
            content += f"| **{m['symbol']}** | {m['signal']} | {wr_str} | {acc_str} | {m['trades']} |\n"
    else:
        content += "| *None yet* | - | - | - | - |\n"

    # Write to file
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    print("Starting Training Monitor...")
    while True:
        try:
            parse_log()
            update_artifact()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(30)
