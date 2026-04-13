import sys
import os
import argparse
from pathlib import Path

# Fix module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core.mt5_connector import get_mt5

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

def close_positions(symbol: str):
    print(f"--- 🔄 Executing Close Logic for {symbol} ---")
    
    _mt5 = get_mt5()
    if not _mt5:
        print("❌ Failed to connect to MT5 Bridge.")
        return False
        
    # Get open positions
    positions = _mt5.positions_get(symbol=symbol)
    
    if not positions:
        print(f"ℹ️ No open positions found for {symbol}")
        return True
        
    print(f"found {len(positions)} open positions for {symbol}")
    
    all_closed = True
    
    for pos in positions:
        ticket = pos.ticket
        pos_type = pos.type # 0 = BUY, 1 = SELL
        vol = pos.volume
        
        action = _mt5.TRADE_ACTION_DEAL
        calc_type = _mt5.ORDER_TYPE_SELL if pos_type == _mt5.ORDER_TYPE_BUY else _mt5.ORDER_TYPE_BUY
        
        tick = _mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"❌ Could not get tick for {symbol}")
            all_closed = False
            continue
            
        price = tick.bid if calc_type == _mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": action,
            "symbol": symbol,
            "volume": vol,
            "type": calc_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Smart Pivot Exit",
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_IOC,
        }
        
        print(f"📦 Closing Ticket {ticket} ({'BUY' if pos_type==0 else 'SELL'} {vol})...")
        result = _mt5.order_send(request)
        
        if result is None:
            print(f"❌ Close Failed: Result is None")
            all_closed = False
            continue

        if result.retcode != _mt5.TRADE_RETCODE_DONE:
            print(f"❌ Close Failed: {result.comment}")
            all_closed = False
        else:
            print(f"✅ Closed Successfully (Ret: {result.retcode})")
    return all_closed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", type=str, help="Symbol to close")
    args = parser.parse_args()
    
    close_positions(args.symbol)
