import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Unicode print issues on Windows
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.notifications import NotificationManager
from core.mt5_connector import get_mt5

def test_pipeline():
    print("=" * 60)
    print("Testing End-to-End Delivery Pipeline")
    print("=" * 60)
    
    # --- 1. TELEGRAM ---
    print("\n[1] Testing Telegram Notification...")
    nm = NotificationManager()
    
    if not nm.enabled:
        print("❌ Telegram is disabled in config.yaml.")
    else:
        mock_data = {
            'symbol': 'EURUSD',
            'signal': 'BUY',
            'confidence': 0.999,
            'price_at_signal': 1.08500,
            'tp_price': 1.08900,
            'sl_price': 1.08200,
            'tp_pips': 40,
            'sl_pips': 30,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regime_threshold': 0.70,
            'suggested_lots': 0.01,
            'model_trades': 1337
        }
        success = nm.send_signal_alert(mock_data)
        if success:
            print("✅ Telegram message sent successfully!")
        else:
            print("❌ Telegram message failed. Check API keys.")

    # --- 2. MT5 ---
    print("\n[2] Testing MT5 Execution...")
    mt5 = get_mt5()
    
    if not mt5:
        print("❌ MT5 Connection Failed.")
        return

    acc = mt5.account_info()
    if acc:
        print(f"✅ Connected to MT5: {acc.server} (Login: {acc.login})")
        print(f"💰 Balance: {acc.balance} {acc.currency}")
    
    # Place a micro 0.01 lot test trade
    symbol = "EURUSD"
    print(f"\nAttempting to place 0.01 BUY on {symbol}...")
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select {symbol}")
        return
        
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"❌ Symbol {symbol} not found")
        return
        
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ Cannot get price for {symbol}")
        return
        
    price = tick.ask
    
    # Determine filling mode
    filling_type = mt5.ORDER_FILLING_FOK
    if (symbol_info.filling_mode & 1):
        filling_type = mt5.ORDER_FILLING_FOK
    elif (symbol_info.filling_mode & 2):
        filling_type = mt5.ORDER_FILLING_IOC
        
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - 0.0020, # 20 pips SL
        "tp": price + 0.0020, # 20 pips TP
        "deviation": 20,
        "magic": 999999,      # Test Magic Number
        "comment": "PIPELINE_TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed! code={result.retcode} comment={result.comment}")
    else:
        print(f"✅ 🟢 Order placed successfully! Ticket: {result.order}")
        
    print("\n=" * 60)
    print("Test Complete.")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
