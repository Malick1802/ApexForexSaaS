import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.executive import ExecutiveEngine

def test_mt5_order():
    print("--- APEX MT5 EXECUTION TEST ---")
    print("Initializing Executive Engine...")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    try:
        exe = ExecutiveEngine()
        
        # Get current price
        from core.mt5_connector import get_mt5
        mt5 = get_mt5()
        if not mt5:
            print("ERROR: MT5 Not connected! Check if terminal/bridge is running.")
            return
            
        symbol = "EURUSD"
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"ERROR: Could not get {symbol} tick info! Ensure symbol is visible in Market Watch.")
            return
            
        price = tick.ask
        
        # ── Mock Signal Data ────────────────────────────────────────────────────────
        # 0.5% risk calculation will happen inside place_mt5_trade automatically
        mock_signal = {
            'symbol': symbol,
            'signal': 'BUY',
            'price_at_signal': price,
            'sl_price': price - 0.0020,  # 20 pips away
            'tp_price': price + 0.0030,  # 30 pips away
            'confidence_tier': 99,       # Max tier
            'confidence': 0.99,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_hidden': 0
        }
        
        print(f"Current {symbol} Price: {price}")
        print(f"Sending Mock Signal: {mock_signal['signal']} {symbol} @ {mock_signal['price_at_signal']}")
        
        # This will trigger the full production logic: Drawdown check -> Lot calculation -> Order Send
        success = exe.place_mt5_trade(mock_signal)
        
        if success:
            print("\n✅ SUCCESS: Mock trade placed successfully in MT5!")
            print("Check your MT5 terminal (Trade tab) to see the new position.")
        else:
            print("\n❌ FAILED: MT5 rejected the order or connection was lost.")
            print("Check 'logs/system_v2.log' for the specific error message.")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_mt5_order()
