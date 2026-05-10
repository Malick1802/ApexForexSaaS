import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import InferenceEngine

def check_current_buy_intent():
    engine = InferenceEngine()
    symbols = engine.data_engine.get_all_pairs()
    if "GOLD" not in symbols: symbols.append("GOLD")
    
    print(f"\nChecking current AI intent for {len(symbols)} pairs using Foundation Brain...\n")
    print(f"{'Symbol':<10} | {'Buy Prob':<10} | {'Sell Prob':<10} | {'Intent':<10}")
    print("-" * 50)
    
    buys_found = []
    
    for symbol in symbols:
        try:
            # We use predict_symbol which now forces Foundation (since we edited inference.py)
            pred = engine.predict_symbol(symbol)
            if pred:
                buy_prob = pred.get('buy_prob', 0)
                sell_prob = pred.get('sell_prob', 0)
                intent = "BUY" if buy_prob > sell_prob and buy_prob > 0.5 else "SELL" if sell_prob > buy_prob and sell_prob > 0.5 else "WAIT"
                
                print(f"{symbol:<10} | {buy_prob:<10.2f} | {sell_prob:<10.2f} | {intent:<10}")
                
                if intent == "BUY":
                    buys_found.append((symbol, buy_prob))
        except Exception as e:
            # print(f"Error checking {symbol}: {e}")
            continue
            
    if buys_found:
        print("\n" + "="*30)
        print("  CURRENT BUY INTENTS FOUND")
        print("="*30)
        for s, p in buys_found:
            print(f"  ✅ {s}: {p:.2%} confidence")
    else:
        print("\n  ❌ No active BUY intents found (Confidence > 50%)")

if __name__ == "__main__":
    check_current_buy_intent()
