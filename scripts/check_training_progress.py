
import os
import time
from pathlib import Path
import json

def check_progress():
    models_dir = Path("models")
    updated_files = []
    now = time.time()
    
    for root, dirs, files in os.walk(models_dir):
        if "config.json" in files:
            path = Path(root) / "config.json"
            mtime = os.path.getmtime(path)
            if now - mtime < 3600: # Last hour
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                    updated_files.append({
                        "path": path,
                        "mtime": mtime,
                        "trades": config.get("trades", 0),
                        "win_rate": config.get("win_rate", 0),
                        "optimized": config.get("optimized", False)
                    })
                except:
                    pass
    
    updated_files.sort(key=lambda x: x['mtime'], reverse=True)
    
    print(f"--- Volume-Optimized Models Updated in Last Hour: {len(updated_files)} ---")
    for f in updated_files:
        rel_path = f['path'].relative_to(models_dir)
        opt_tag = "[OPTIMIZED]" if f['optimized'] else "[BASE]"
        print(f"{opt_tag} {rel_path}: {f['trades']} trades at {f['win_rate']:.1%} WR")

if __name__ == "__main__":
    check_progress()
