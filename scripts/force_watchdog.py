
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.core.database import SignalDatabase
from data_pipeline import DataEngine
import pandas as pd

def force_check():
    db = SignalDatabase()
    engine = DataEngine()
    
    active = db.get_active_signals()
    print(f"Checking {len(active)} active signals...")
    
    for sig in active:
        symbol = sig['symbol']
        print(f"  Scanning {symbol} ({sig['signal']})...")
        
        # Fetch detailed 1m data
        df = engine.fetch(symbol, interval="1m", days=1, use_cache=False)
        if df.empty:
            print("    No data.")
            continue
            
        # Filter data since signal timestamp
        sig_ts = pd.to_datetime(sig['timestamp'])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.tz_localize('UTC')
            
        # Ensure df index is timezone aware (UTC)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
            
        # Get candles AFTER signal was generated
        relevant_data = df[df.index >= sig_ts]
        
        if relevant_data.empty:
            print("    No data since signal.")
            continue
            
        tp = sig.get('tp_price')
        sl = sig.get('sl_price')
        direction = sig['signal']
        sig_id = sig['id']
        curr_price = df['close'].iloc[-1]
        
        outcome = None
        hit_ts = None
        
        if direction == 'BUY':
            # Check SL (Low) - Any candle hitting SL?
            if sl:
                sl_hits = relevant_data[relevant_data['low'] <= sl]
                if not sl_hits.empty:
                    outcome = 'FAIL'
                    hit_ts = sl_hits.index[0]
                    print(f"    ❌ HIT SL! Low {sl_hits['low'].iloc[0]} <= {sl} at {hit_ts}")
            
            # Check TP (High) - Any candle hitting TP?
            if not outcome and tp:
                tp_hits = relevant_data[relevant_data['high'] >= tp]
                if not tp_hits.empty:
                    # Check if TP happened BEFORE SL?
                    outcome = 'SUCCESS'
                    hit_ts = tp_hits.index[0]
                    print(f"    🎯 HIT TP! High {tp_hits['high'].iloc[0]} >= {tp} at {hit_ts}")
                    
        elif direction == 'SELL':
            # Check SL (High)
            if sl:
                sl_hits = relevant_data[relevant_data['high'] >= sl]
                if not sl_hits.empty:
                    outcome = 'FAIL'
                    hit_ts = sl_hits.index[0]
                    print(f"    ❌ HIT SL! High {sl_hits['high'].iloc[0]} >= {sl} at {hit_ts}")

            # Check TP (Low)
            if not outcome and tp:
                tp_hits = relevant_data[relevant_data['low'] <= tp]
                if not tp_hits.empty:
                    outcome = 'SUCCESS'
                    hit_ts = tp_hits.index[0]
                    print(f"    🎯 HIT TP! Low {tp_hits['low'].iloc[0]} <= {tp} at {hit_ts}")

        if outcome:
            db.update_signal_outcome(sig_id, outcome)
            print(f"    Updated DB to {outcome}")
        else:
            print(f"    Still active. Price: {curr_price}, SL: {sl}, TP: {tp}")

if __name__ == "__main__":
    force_check()
