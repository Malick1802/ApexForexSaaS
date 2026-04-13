
import sqlite3
import sys
import os
from pathlib import Path

# Fix module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mt5_connector import get_mt5

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

def execute_active_trade():
    print("--- 🚀 Executing Active Trade on MT5 ---")
    
    # 1. Get Active Signal from DB
    conn = sqlite3.connect("signals.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Filter for actionable signals only
    cursor.execute("SELECT * FROM signals WHERE outcome='ACTIVE' AND signal IN ('BUY', 'SELL') ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("❌ No ACTIVE BUY/SELL signals found to trade.")
        return

    sig = dict(row)
    symbol = sig['symbol']
    signal = sig['signal'] # BUY or SELL
    sl = sig['sl_price']
    tp = sig['tp_price']
    
    print(f"🎯 Target Signal: {symbol} {signal}")
    print(f"   SL: {sl} | TP: {tp}")

    # 2. Connect to MT5
    _mt5 = get_mt5()
    if not _mt5:
        print("❌ Failed to connect to MT5 Bridge.")
        return

    # Check if symbol is visible in Market Watch (Required for trading)
    selected = _mt5.symbol_select(symbol, True)
    if not selected:
        print(f"❌ Failed to select {symbol} in MT5")
        return
    
    # Check filling mode
    filling_type = _mt5.ORDER_FILLING_FOK # Default
    symbol_info = _mt5.symbol_info(symbol)
    if symbol_info:
        # Check flags (1=FOK, 2=IOC)
        # If IOC (2) is supported, use it. Else FOK.
        if (symbol_info.filling_mode & 2) != 0:
            filling_type = _mt5.ORDER_FILLING_IOC
        elif (symbol_info.filling_mode & 1) != 0:
            filling_type = _mt5.ORDER_FILLING_FOK
            
    print(f"⚙️ Using Filling Mode: {filling_type}")
    
    # Re-define variables
    action = _mt5.TRADE_ACTION_DEAL
    order_type = _mt5.ORDER_TYPE_BUY if signal == 'BUY' else _mt5.ORDER_TYPE_SELL
    
    tick = _mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ Could not get tick for {symbol}")
        return
        
    price = tick.ask if signal == 'BUY' else tick.bid
    deviation = 20

    request = {
        "action": action,
        "symbol": symbol,
        "volume": 0.01, # Tiny test lot
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 123456,
        "comment": "ApexForex Auto-Test",
        "type_time": _mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    print("\n📦 Sending Order Request:")
    print(request)
    
    # 4. Execute
    result = _mt5.order_send(request)
    
    if result is None:
        print("❌ Order Send Result is None (Bridge failure?)")
        return

    print(f"\n📨 Order Send Result: retcode={result.retcode}")
    if result.retcode != _mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order Failed: {result.comment}")
    else:
        print(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        print(f"   Ticket: {result.order}")
        print(f"   Price: {result.price}")
        print(f"   Comment: {result.comment}")

    # Initialization handled by singleton, no manual shutdown here

if __name__ == "__main__":
    execute_active_trade()
