import pandas as pd
import numpy as np
from core.regime_detector import MarketRegime

# Proposed Relaxed Thresholds
NEW_EMA_STRETCH = 5.0
NEW_RSI_HIGH = 90.0
NEW_RSI_LOW = 10.0
NEW_ATR_Z = 4.0

def test_thresholds(symbol="EURUSD"):
    from data_pipeline import DataEngine
    engine = DataEngine()
    df = engine.fetch(symbol, interval="1h", days=30)
    if df.empty: return
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 1. ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().dropna()
    mu, sigma = atr.tail(100).mean(), atr.tail(100).std()
    atr_z = (atr.iloc[-1] - mu) / sigma if sigma > 0 else 0
    
    # 2. EMA Stretch
    ema200 = close.ewm(span=200, adjust=False).mean()
    dist = abs(close.iloc[-1] - ema200.iloc[-1])
    atr_val = atr.iloc[-1]
    stretch = dist / atr_val if atr_val > 0 else 0
    
    # 3. RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50.0
    
    print(f"--- {symbol} Metrics ---")
    print(f"  ATR Z-Score: {atr_z:.2f} (Old Limit: 3.5 | New: {NEW_ATR_Z})")
    print(f"  EMA Stretch: {stretch:.2f} ATRs (Old Limit: 2.5 | New: {NEW_EMA_STRETCH})")
    print(f"  RSI:         {rsi:.1f} (Old Limit: 80/20 | New: {NEW_RSI_HIGH}/{NEW_RSI_LOW})")
    
    is_crisis_old = (atr_z >= 3.5 or stretch >= 2.5 or rsi >= 80 or rsi <= 20)
    is_crisis_new = (atr_z >= NEW_ATR_Z or stretch >= NEW_EMA_STRETCH or rsi >= NEW_RSI_HIGH or rsi <= NEW_RSI_LOW)
    
    print(f"\n  Old Logic: {'🚨 CRISIS' if is_crisis_old else '✅ OK'}")
    print(f"  New Logic: {'🚨 CRISIS' if is_crisis_new else '✅ OK'}")

if __name__ == "__main__":
    import os
    os.environ['PYTHONPATH'] = "."
    for s in ["EURUSD", "GBPUSD", "USDCHF", "AUDUSD"]:
        test_thresholds(s)
