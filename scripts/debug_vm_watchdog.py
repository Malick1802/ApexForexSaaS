import sqlite3
import pandas as pd
from datetime import datetime, timezone
import sys
import os

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.executive import ExecutiveEngine

def debug_active_signals():
    print("Initializing Engine & Fetching MT5 Connection...")
    engine = ExecutiveEngine()
    
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get all active BUY/SELL signals
    cur.execute('''
        SELECT id, symbol, signal, confidence, outcome, timestamp, tp_price, sl_price 
        FROM signals 
        WHERE outcome='ACTIVE' AND signal IN ('BUY','SELL')
        ORDER BY timestamp DESC
    ''')
    active_signals = cur.fetchall()
    
    if not active_signals:
        print("\n✅ ZERO active BUY/SELL signals in the database.")
        print("If your dashboard shows active signals, it is a UI caching issue. Force-refresh the browser.")
        return

    print(f"\n🔍 Found {len(active_signals)} ACTIVE signals in DB. Analyzing against live MT5 data...\n")
    
    for row in active_signals:
        sig_id = row['id']
        symbol = row['symbol']
        direction = row['signal']
        tp = row['tp_price']
        sl = row['sl_price']
        sig_ts_str = row['timestamp']
        
        print(f"[{symbol}] {direction} (ID: {sig_id}) | Generated: {sig_ts_str[:19]} | TP: {tp} | SL: {sl}")
        
        # 1. Check valid TP/SL
        if not tp or not sl or tp == 0.0 or sl == 0.0:
            print("  ❌ ERROR: Signal has missing or zero TP/SL levels. Watchdog will mark as EXPIRED.")
            continue
            
        # 2. Parse timestamp
        try:
            sig_ts = pd.to_datetime(sig_ts_str)
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.tz_localize('UTC')
        except Exception as e:
            print(f"  ❌ ERROR: Failed to parse timestamp: {e}")
            continue
            
        # 3. Fetch MT5 Data
        df = engine.inference_engine.data_engine.fetch(symbol, interval="1m", days=3, use_cache=False)
        if df.empty:
            print(f"  ❌ ERROR: MT5 returned EMPTY data for {symbol}. Bridge disconnected or symbol invalid.")
            continue
            
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
            
        relevant = df[df.index >= sig_ts]
        if relevant.empty:
            print(f"  ⚠️ WARNING: No MT5 candles found *after* the signal time. Data feed might be lagging.")
            print(f"     Last candle available: {df.index[-1]}")
            continue
            
        # 4. Check prices
        max_high = relevant['high'].max()
        min_low = relevant['low'].min()
        print(f"  📊 MT5 Data since signal -> Max High: {max_high:.5f} | Min Low: {min_low:.5f}")
        
        outcome = None
        if direction == 'BUY':
            if (relevant['low'] <= sl).any(): outcome = 'FAIL (SL Hit)'
            elif (relevant['high'] >= tp).any(): outcome = 'SUCCESS (TP Hit)'
            
            pips_to_tp = (tp - max_high) * 10000
            if outcome is None: print(f"  ⏳ Status: Still active. Needs to rise another {pips_to_tp:.1f} pips to hit TP.")
            
        elif direction == 'SELL':
            if (relevant['high'] >= sl).any(): outcome = 'FAIL (SL Hit)'
            elif (relevant['low'] <= tp).any(): outcome = 'SUCCESS (TP Hit)'
            
            pips_to_tp = (min_low - tp) * 10000
            if outcome is None: print(f"  ⏳ Status: Still active. Needs to fall another {pips_to_tp:.1f} pips to hit TP.")
            
        if outcome:
            print(f"  ✅ WATCHDOG VERDICT: Should be marked as {outcome}")
        print("-" * 60)

    conn.close()

if __name__ == "__main__":
    debug_active_signals()
