import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core.notifications import NotificationManager

def test_prop_precision_alert():
    print("🚀 Initializing Prop-Grade Mock Alert (0.5% Risk)...")
    
    # Connect to MT5 for real-time diagnostic
    from core.core.mt5_connector import get_mt5
    _mt5 = get_mt5()
    if not _mt5:
        print("❌ MT5 Bridge failed. Falling back to static mock.")
        equity = 100000.0
    else:
        acc = _mt5.account_info()
        equity = acc.equity
        print(f"💰 Real-Time Equity Detected: ${equity:,.2f}")

    notifier = NotificationManager()
    
    if not notifier.enabled:
        print("❌ Telegram is DISABLED. Check config.yaml.")
        return

    # Simulation of a Gold Trade (GOLD)
    # Entry: 2402.10, SL: 2395.20 (6.90 distance)
    # Target Risk: 0.5% of Equity
    risk_pct = 0.5
    risk_amount = equity * (risk_pct / 100.0)
    
    symbol = "GOLD"
    entry = 2402.10
    sl = 2395.20
    price_dist = abs(entry - sl)
    
    if _mt5:
        sym_info = _mt5.symbol_info("GOLD") or _mt5.symbol_info("XAUUSD")
        if sym_info:
            tick_size = sym_info.trade_tick_size
            tick_value = sym_info.trade_tick_value
            dist_in_ticks = price_dist / tick_size
            loss_per_lot = dist_in_ticks * tick_value
            lots = risk_amount / loss_per_lot
            
            # Normalize
            step = sym_info.volume_step
            lots = round(lots / step) * step
            symbol = sym_info.name
        else:
            lots = 0.72 # Fallback
    else:
        lots = 0.72 # Fallback

    mock_data = {
        'symbol': symbol,
        'signal': 'BUY',
        'confidence': 0.884,
        'price_at_signal': entry,
        'tp_price': 2415.50,
        'sl_price': sl,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'regime_threshold': 0.70,
        'suggested_lots': round(lots, 2),
        'model_trades': 1250
    }

    print(f"📡 Sending Precision Alert for {mock_data['symbol']} (Lots: {mock_data['suggested_lots']})...")
    success = notifier.send_signal_alert(mock_data)
    
    if success:
        print("✅ SUCCESS: Precision Alert Sent. Check your Telegram!")
        print(f"📊 Displayed Data: Risk=0.5% | Lots={mock_data['suggested_lots']}")
    else:
        print("❌ FAILED to send alert. Check logs or .env credentials.")

if __name__ == "__main__":
    test_prop_precision_alert()
