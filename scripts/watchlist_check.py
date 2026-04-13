import os
import sys
import yaml
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from core.core.inference import InferenceEngine

def get_watchlist():
    print("🧠 Initializing Inference Engine...")
    engine = InferenceEngine()
    
    # Get all symbols from config
    symbols = []
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        for cat in ["majors", "minors", "crosses"]:
            for p in config.get("currency_pairs", {}).get(cat, []):
                symbols.append(p["symbol"])
                
    print(f"Checking {len(symbols)} pairs for Near-Miss setups (60-69%)...")
    
    watchlist = []
    for s in symbols:
        try:
            # Run prediction without saving to DB
            res = engine.predict_symbol(s, win_rate="60%", save_to_db=False)
            if res and res.get("confidence", 0) >= 0.60:
                conf = res["confidence"]
                if conf < 0.70:
                    watchlist.append({
                        "symbol": s,
                        "signal": res["signal"],
                        "confidence": conf,
                        "status": "NEAR-MISS"
                    })
                else:
                    watchlist.append({
                        "symbol": s,
                        "signal": res["signal"],
                        "confidence": conf,
                        "status": "VALID-SIGNAL"
                    })
        except Exception as e:
            pass
            
    # Sort by confidence descending
    watchlist.sort(key=lambda x: x["confidence"], reverse=True)
    
    print("\n--- INSTITUTIONAL WATCHLIST (> 60% Conviction) ---")
    if not watchlist:
        print("No pairs currently above 60% hurdle.")
    else:
        for r in watchlist:
            label = "🔥 SIGNAL" if r["status"] == "VALID-SIGNAL" else "👀 WATCH"
            print(f"{label} | {r['symbol']}: {r['signal']} ({r['confidence']:.1%})")
    print("-" * 50)

if __name__ == "__main__":
    get_watchlist()
