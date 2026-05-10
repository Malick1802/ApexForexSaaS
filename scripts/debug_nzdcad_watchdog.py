import sqlite3
import pandas as pd
from core.executive import ExecutiveEngine
import logging

logging.basicConfig(level=logging.INFO)
engine = ExecutiveEngine()
conn = sqlite3.connect('signals.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('''SELECT id, symbol, signal, timestamp, tp_price, sl_price FROM signals WHERE symbol='NZDCAD' AND outcome='ACTIVE' ORDER BY timestamp DESC LIMIT 1''')
row = cur.fetchone()
if not row:
    print('No active NZDCAD signal found in DB.')
else:
    print(f'Signal TS in DB: {row["timestamp"]}')
    sig_ts = pd.to_datetime(row['timestamp'])
    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.tz_localize('UTC')
    print(f'Parsed Sig TS (UTC): {sig_ts}')
    
    df = engine.inference_engine.data_engine.fetch('NZDCAD', interval='1m', days=2, use_cache=False)
    if df.empty:
        print('MT5 returned empty DF')
    else:
        print(f'MT5 Last Candle Time: {df.index[-1]}')
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
            
        print(f'MT5 Adjusted Last Candle: {df.index[-1]}')
        
        relevant = df[df.index >= sig_ts]
        print(f'Relevant Candles Count: {len(relevant)}')
        if not relevant.empty:
            print(f'Relevant range: {relevant.index[0]} to {relevant.index[-1]}')
            
            # Print High/Low vs TP/SL
            tp = row['tp_price']
            sl = row['sl_price']
            print(f'TP: {tp}, SL: {sl}')
            if row['signal'] == 'BUY':
                print(f"Max High: {relevant['high'].max()} >= TP? {relevant['high'].max() >= tp}")
            else:
                print(f"Min Low: {relevant['low'].min()} <= TP? {relevant['low'].min() <= tp}")
conn.close()
