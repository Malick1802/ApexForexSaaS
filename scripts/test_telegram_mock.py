import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.notifications import NotificationManager

def send_mock_alert():
    print("🚀 Initializing Mock Telegram Alert...")
    notifier = NotificationManager()
    
    if not notifier.enabled:
        print("❌ Telegram is currently DISABLED in config.yaml.")
        print("Please set notifications -> telegram -> enabled: true first.")
        return

    # Mock Signal Data (High Fidelity)
    mock_data = {
        'symbol': 'XAUUSD (GOLD)',
        'signal': 'BUY',
        'confidence': 0.845,
        'model_trades': 1250,
        'price_at_signal': 2402.10,
        'tp_price': 2415.50,
        'sl_price': 2395.20,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'regime_threshold': 0.70  # Explicitly set to pass the internal hurdle check
    }

    print(f"📡 Sending mock signal for {mock_data['symbol']}...")
    success = notifier.send_signal_alert(mock_data)
    
    if success:
        print("✅ Mock Alert Sent! Check your Telegram.")
    else:
        print("❌ Failed to send alert. Check logs/system_v2.log or your .env credentials.")

if __name__ == "__main__":
    send_mock_alert()
