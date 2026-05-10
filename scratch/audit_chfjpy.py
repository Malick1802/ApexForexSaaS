import sys
import os
import json
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

def analyze_chfjpy():
    engine = InferenceEngine()
    symbol = "CHFJPY"
    
    print(f"\n--- Deep Audit: {symbol} ---")
    
    # Run full prediction loop to get diagnostics
    # This will use the Foundation Brain (since we forced it in inference.py)
    pred = engine.predict_symbol(symbol)
    
    if not pred:
        print(f"Failed to get prediction for {symbol}")
        return

    print(f"AI Intent: {pred.get('signal')}")
    print(f"Confidence: {pred.get('confidence'):.2%}")
    print(f"Buy Prob: {pred.get('buy_prob', 0):.2%}")
    print(f"Sell Prob: {pred.get('sell_prob', 0):.2%}")
    print(f"Wait Prob: {pred.get('wait_prob', 0):.2%}")
    
    # Check features that might influence this
    # We can fetch the raw data and features
    df = engine.data_engine.fetch(symbol, interval="1h", days=5)
    if df.empty:
        print("No price data found.")
        return
        
    last_price = df['close'].iloc[-1]
    print(f"Current Price: {last_price}")
    
    # Calculate some basics
    from data_pipeline.features import FeatureEngineer
    fe = FeatureEngineer()
    features = fe.extract_features(df)
    
    last_features = features.iloc[-1]
    print("\nKey Indicators:")
    if 'rsi' in last_features:
        print(f"  RSI: {last_features['rsi']:.2f}")
    if 'bb_position' in last_features:
        print(f"  BB Position: {last_features['bb_position']:.2f} (1.0 = Upper Band)")
    if 'macd_hist' in last_features:
        print(f"  MACD Hist: {last_features['macd_hist']:.4f}")
        
    print("\nConclusion: ")
    if pred.get('sell_prob', 0) > pred.get('buy_prob', 0):
        if last_features.get('rsi', 0) > 70:
            print("  -> Overbought RSI detected. AI is likely looking for a mean-reversion reversal.")
        elif last_features.get('bb_position', 0) > 1.0:
            print("  -> Price is above the upper Bollinger Band. AI is likely anticipating a pullback.")
        else:
            print("  -> AI sees a structural resistance or a macro-context (JPY strength) not obvious on the 1h chart alone.")

if __name__ == "__main__":
    analyze_chfjpy()
