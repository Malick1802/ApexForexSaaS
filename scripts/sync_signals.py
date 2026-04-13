
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core.database import SignalDatabase
from data_pipeline import DataEngine

def sync_active():
    db = SignalDatabase()
    de = DataEngine()
    signals = db.get_active_signals()
    
    if not signals:
        print("No active signals to sync.")
        return

    for s in signals:
        symbol = s['symbol']
        try:
            df = de.fetch(symbol, interval='1m', days=2, use_cache=False)
            if df.empty: continue
            
            sig_ts = pd.to_datetime(s['timestamp'])
            if sig_ts.tzinfo is None: sig_ts = sig_ts.tz_localize('UTC')
            if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
            
            relevant = df[df.index >= sig_ts]
            if relevant.empty: continue
            
            tp = s['tp_price']
            sl = s['sl_price']
            direction = s['signal']
            
            outcome = None
            if direction == 'BUY':
                if (relevant['high'] >= tp).any(): outcome = 'SUCCESS'
                elif (relevant['low'] <= sl).any(): outcome = 'FAIL'
            else: # SELL
                if (relevant['low'] <= tp).any(): outcome = 'SUCCESS'
                elif (relevant['high'] >= sl).any(): outcome = 'FAIL'
                
            if outcome:
                print(f"Syncing {symbol} {direction} to {outcome}")
                db.update_signal_outcome(s['id'], outcome)
        except Exception as e:
            print(f"Error syncing {symbol}: {e}")

if __name__ == "__main__":
    sync_active()
