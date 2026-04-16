import logging
import sqlite3
import pandas as pd
from core.executive import ExecutiveEngine

logging.basicConfig(level=logging.INFO)

def debug_nzjpy_resolution():
    engine = ExecutiveEngine()
    
    conn = sqlite3.connect('signals.db')
    cursor = conn.cursor()
    # Fetch exactly one of the NZDJPY ghost signals
    cursor.execute("SELECT * FROM signals WHERE id = 26150")
    cols = [description[0] for description in cursor.description]
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("Signal 26150 not found.")
        return
        
    sig = dict(zip(cols, row))
    print(f"Debug Signal: {sig}")
    
    symbol = sig['symbol']
    # Fetch detailed 1m data (Lookback 14 days)
    df = engine.inference_engine.data_engine.fetch(symbol, interval="1m", days=14, use_cache=False)
    if df.empty:
        print("Data is empty.")
        return
        
    # Filter since signal time
    sig_ts = pd.to_datetime(sig['timestamp'])
    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.tz_localize('UTC')
    
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
        
    relevant = df[df.index >= sig_ts]
    
    if relevant.empty:
        print("Relevant data is empty. Issue with timestamps!")
        print(f"Signal TS: {sig_ts}, Data Start: {df.index.min()}, Data End: {df.index.max()}")
        return
    
    tp = sig.get('tp_price')
    sl = sig.get('sl_price')
    direction = sig['signal']
    outcome = None
    
    print(f"Checking {direction}. TP: {tp}, SL: {sl}")
    print(f"Relevant Min Low: {relevant['low'].min()}, Max High: {relevant['high'].max()}")
    
    if direction == 'BUY':
        # Check SL (Low)
        if sl and (relevant['low'] <= sl).any():
            outcome = 'FAIL'
            print(f"FAIL: {symbol} hit SL {sl}")
        # Check TP (High)
        elif tp and (relevant['high'] >= tp).any():
            outcome = 'SUCCESS'
            print(f"SUCCESS: {symbol} hit TP {tp}")
            
    elif direction == 'SELL':
        pass
        
    if outcome:
        print(f"Outcome determined: {outcome}")
    else:
        print("No outcome determined, stays ACTIVE.")

if __name__ == "__main__":
    debug_nzjpy_resolution()
