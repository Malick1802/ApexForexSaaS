import pandas as pd
import numpy as np
from core.regime_detector import RegimeDetector
from data_pipeline import DataEngine

def diagnose_crisis(symbol="EURUSD"):
    print(f"=== Diagnosing CRISIS for {symbol} ===")
    
    # 1. Fetch Data
    engine = DataEngine()
    df = engine.fetch(symbol, interval="1h", days=30)
    if df.empty:
        print("No data found.")
        return

    # 2. Run Detector
    detector = RegimeDetector()
    results = detector.detect(df, symbol)
    
    if results:
        print(f"\nDetection Results:")
        print(f"  Regime: {results.regime}")
        print(f"  Volatility (ATR): {results.volatility:.6f}")
        # We need to see the internal Z-scores. I'll hack into the detector or re-calculate.
        
        # Manual ATR Z-Score Check (matching core/regime_detector.py logic)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        atr_lookback = atr.tail(100) # ZSCORE_LOOKBACK
        mean_atr = atr_lookback.mean()
        std_atr = atr_lookback.std()
        current_atr = atr.iloc[-1]
        
        z_score = (current_atr - mean_atr) / std_atr if std_atr > 0 else 0
        
        print(f"\nInternal Metrics:")
        print(f"  Current ATR: {current_atr:.6f}")
        print(f"  Mean ATR (100 bars): {mean_atr:.6f}")
        print(f"  Std ATR (100 bars): {std_atr:.6f}")
        print(f"  ATR Z-Score: {z_score:.2f}")
        print(f"  Crisis Threshold: 3.5") # My current threshold

        if z_score > 3.5:
             print("\n!!! CONFIRMED: ATR Z-Score triggers CRISIS")

if __name__ == "__main__":
    import os
    os.environ['PYTHONPATH'] = "."
    diagnose_crisis("EURUSD")
