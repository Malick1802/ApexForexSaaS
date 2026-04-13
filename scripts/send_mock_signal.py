import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.notifications import NotificationManager

def send_mock_notification():
    print("🚀 Sending Mock Restoration Notification...")
    nm = NotificationManager()
    
    if not nm.enabled:
        print("❌ Telegram is disabled. Check config.yaml.")
        return

    mock_data = {
        'symbol': 'EURUSD',
        'signal': 'BUY',
        'confidence': 0.892,
        'price_at_signal': 1.08542,
        'tp_price': 1.08912,
        'sl_price': 1.08292,
        'tp_pips': 37,
        'sl_pips': 25,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'regime_threshold': 0.70,
        'suggested_lots': 0.12,
        'model_trades': 842
    }

    success = nm.send_signal_alert(mock_data)
    if success:
        print("✅ Mock notification sent successfully!")
    else:
        print("❌ Failed to send mock notification.")

if __name__ == "__main__":
    send_mock_notification()
