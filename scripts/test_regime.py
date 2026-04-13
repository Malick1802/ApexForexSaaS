import sys; sys.path.insert(0, '.')
from data_pipeline.engine import DataEngine
from core.core.regime_detector import RegimeDetector

engine = DataEngine()
detector = RegimeDetector()
pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDCAD', 'NZDJPY', 'NZDCHF', 'AUDUSD', 'USDCAD']
rows = []
for sym in pairs:
    df = engine.fetch(sym, interval='1h', days=30)
    if df is not None:
        r = detector.detect(df, sym)
        if r:
            rows.append((sym, r.regime.value, r.adx, r.atr_zscore, r.bb_zscore, r.confidence_threshold, r.block_trading))

print(f"{'Symbol':<10} {'Regime':<16} {'ADX':>6} {'ATR_z':>7} {'BB_z':>6} {'Threshold':>10} {'Block':>6}")
print('-'*66)
for sym, regime, adx, atr_z, bb_z, thr, block in rows:
    print(f"{sym:<10} {regime:<16} {adx:>6.1f} {atr_z:>7.2f} {bb_z:>6.2f} {thr:>10.0%} {'YES' if block else 'no':>6}")
