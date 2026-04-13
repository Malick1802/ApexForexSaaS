import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.engine import DataEngine
from core.regime_detector import get_detector

def run_regime_check(all_pairs=False):
    engine = DataEngine()
    detector = get_detector()
    
    if all_pairs:
        pairs = engine.get_all_pairs()
    else:
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDCAD', 'NZDJPY', 'AUDUSD', 'USDCAD']
    
    print(f"{'Symbol':<10} {'Regime':<16} {'ADX':>6} {'ATR_z':>7} {'BB_z':>6} {'Threshold':>10} {'Block':>6}")
    print('-'*66)
    
    for sym in pairs:
        try:
            df = engine.fetch(sym, interval='1h', days=30)
            if df is not None and not df.empty:
                # is_tradeable returns (bool, float, MarketRegimeResult)
                tradeable, threshold, result = detector.is_tradeable(df, sym)
                
                regime = result.regime.value
                adx = result.adx
                atr_z = result.atr_zscore
                bb_z = result.bb_zscore
                block = not tradeable
                
                print(f"{sym:<10} {regime:<16} {adx:>6.1f} {atr_z:>7.2f} {bb_z:>6.2f} {threshold:>10.0%} {'YES' if block else 'no':>6}")
        except Exception as e:
            print(f"{sym:<10} {'ERROR':<16} {'--':>6} {'--':>7} {'--':>6} {'--':>10} {'--':>6}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    
    run_regime_check(all_pairs=args.all_pairs)
