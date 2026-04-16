import pandas as pd
import numpy as np
from core.regime_detector import RegimeDetector, MarketRegime

def test_regime():
    print("Testing RegimeDetector Stabilization...")
    detector = RegimeDetector()
    
    # 1. Test with insufficient data
    print("Test 1: Insufficient data...", end=" ")
    res = detector.detect(pd.DataFrame())
    print("PASS (Returns None)" if res is None else "FAIL")
    
    # 2. Test with mock data (Normal)
    print("Test 2: Normal data...", end=" ")
    df = pd.DataFrame({
        'close': np.linspace(100, 105, 200),
        'high': np.linspace(101, 106, 200),
        'low': np.linspace(99, 104, 200)
    })
    try:
        res = detector.detect(df, "TEST")
        print(f"PASS (Regime: {res.regime.value if res else 'None'})")
    except Exception as e:
        print(f"FAIL: {e}")

    # 3. Test with "Crisis" data (Volatility spike)
    print("Test 3: Crisis data spike...", end=" ")
    df_crash = df.copy()
    # Add a massive spike at the end
    df_crash.loc[199, 'high'] = 200
    df_crash.loc[199, 'low'] = 50
    df_crash.loc[199, 'close'] = 120
    try:
        res = detector.detect(df_crash, "CRASH_TEST")
        print(f"PASS (Regime: {res.regime.value if res else 'None'})")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    test_regime()
