import logging
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from data_pipeline import DataEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Diagnose")

def check_pair(symbol):
    print(f"\n--- Diagnosing {symbol} ---")
    engine = DataEngine()
    
    # Fetch data
    try:
        # Fetch with default labeling (Tripple Barrier)
        df = engine.fetch_labeled(symbol, interval="1h", days=365)
        
        if df.empty:
            print("❌ No data found.")
            return

        print(f"Data shape: {df.shape}")
        
        # Check Labels
        if 'label' not in df.columns:
            print("❌ 'label' column missing!")
            return
            
        print("\nLabel Distribution:")
        counts = df['label'].value_counts().sort_index()
        total = len(df)
        
        for label, count in counts.items():
            name = {0: "WAIT", 1: "BUY", 2: "SELL"}.get(label, str(label))
            print(f"  {name} ({label}): {count} ({count/total:.2%})")
            
        # Check Features
        print("\nFeature Check:")
        missing = df.isnull().sum().sum()
        print(f"  Missing values: {missing}")
        if missing > 0:
            print(df.isnull().sum()[df.isnull().sum() > 0])
            
        # Check volatility (ATR)
        # We need to manually calculate if not present, but DataEngine might have it?
        # DataEngine returns labeled data, which might NOT have features yet unless we call FeatureEngineer.
        # But let's check price movement vs barriers.
        
        # Estimate typical 1H range
        df['range'] = df['high'] - df['low']
        avg_range = df['range'].mean()
        print(f"\nAvg 1H Range (pips approx): {avg_range:.5f}")
        
        # Check how often we hit 25 pips (typical SL) or 50 pips (typical TP)
        # This is rough, but helps see if barriers are realistic.
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pair("USDJPY")
    check_pair("EURUSD")
