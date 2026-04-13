import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.core.database import get_db
from core.data_engine import DataEngine
import pandas as pd
from datetime import datetime

db = get_db()
engine = DataEngine()

symbol = "EURAUD"

# 1. Get active signal
active = db.get_active_signals(symbol=symbol)
print(f"--- Active Signal for {symbol} ---")
if active:
    sig = active[0]
    print(f"Signal ID: {sig['id']}")
    print(f"Direction: {sig['signal']}")
    print(f"Entry: {sig['price_at_signal']}")
    print(f"SL: {sig.get('sl_price')}")
    print(f"TP: {sig.get('tp_price')}")
    print(f"Timestamp: {sig['timestamp']}")
    
    # 2. Check current price
    df = engine.fetch(symbol, interval="1h", days=5)
    if not df.empty:
        last_close = df['close'].iloc[-1]
        last_low = df['low'].iloc[-1]
        last_high = df['high'].iloc[-1]
        last_time = df.index[-1]
        print(f"\n--- Current Market Data ({last_time}) ---")
        print(f"Close: {last_close}")
        print(f"Low: {last_low}")
        print(f"High: {last_high}")
        
        # Check SL condition
        sl_price = sig.get('sl_price')
        if sl_price:
            sl_hit = False
            if sig['signal'] == 'BUY':
                if last_low <= sl_price:
                    sl_hit = True
                    print(f"🚨 SL HIT! Low {last_low} <= SL {sl_price}")
            elif sig['signal'] == 'SELL':
                if last_high >= sl_price:
                    sl_hit = True
                    print(f"🚨 SL HIT! High {last_high} >= SL {sl_price}")
            
            if not sl_hit:
                print("✅ SL not hit yet.")
        else:
            print("⚠️ No SL price defined in signal.")
            
else:
    print("No active signal found.")
