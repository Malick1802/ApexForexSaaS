
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
import os
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.database import SignalDatabase
from data_pipeline import DataEngine

def check_prices():
    db = SignalDatabase()
    de = DataEngine()
    signals = db.get_active_signals()
    
    if not signals:
        print("No active signals found.")
        return

    print(f"{'Symbol':<10} {'Signal':<7} {'Entry':<10} {'TP':<10} {'SL':<10} {'Current':<10} {'Status'}")
    print("-" * 75)
    
    for s in signals:
        symbol = s['symbol']
        try:
            df = de.fetch(symbol, interval='1m', days=1, use_cache=False)
            if df.empty:
                print(f"{symbol:<10} No data")
                continue
            
            current_price = df.iloc[-1]['close']
            tp = s['tp_price']
            sl = s['sl_price']
            direction = s['signal']
            
            # Check since signal time
            sig_ts = pd.to_datetime(s['timestamp'])
            if sig_ts.tzinfo is None: sig_ts = sig_ts.tz_localize('UTC')
            if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
            relevant = df[df.index >= sig_ts]
            
            hit_tp = False
            hit_sl = False
            
            if direction == 'BUY':
                if (relevant['high'] >= tp).any(): hit_tp = True
                if (relevant['low'] <= sl).any(): hit_sl = True
            else: # SELL
                if (relevant['low'] <= tp).any(): hit_tp = True
                if (relevant['high'] >= sl).any(): hit_sl = True
                
            status = "ACTIVE"
            if hit_tp: status = "SHOULD BE SUCCESS"
            if hit_sl: status = "SHOULD BE FAIL"
            
            print(f"{symbol:<10} {direction:<7} {s['price_at_signal']:<10.5f} {tp:<10.5f} {sl:<10.5f} {current_price:<10.5f} {status}")
        except Exception as e:
            print(f"{symbol:<10} Error: {e}")

if __name__ == "__main__":
    check_prices()
