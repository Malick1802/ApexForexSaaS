import logging
from core.notifications import NotificationManager

# 1. Setup Mock for logic test
# We set up a fake signal_data for a benched model (Gold) 
# currently in a TRENDING market (65% hurdle)
signal_data = {
    'symbol': 'XAUUSD',
    'signal': 'BUY',
    'confidence': 0.62,
    'regime_threshold': 0.65, # Strict regime hurdle
    'is_shadow_alert': True,   # This is a shadow trade
    'price_at_signal': 2345.67,
    'timestamp': '2026-04-16T11:19:00Z',
    'tp_price': 2355.0,
    'sl_price': 2335.0,
    'tp_pips': 100,
    'sl_pips': 100,
    'model_trades': 5
}

# 2. Test Logic
import unittest.mock
with unittest.mock.patch('core.notifications.DataEngine'):
    notifier = NotificationManager()
    # Mocking the dictionary config directly
    notifier.telegram_config = {'alert_threshold': 0.60, 'notify_shadow_trades': True}
    notifier.enabled = True 

print(f"--- TEST: Shadow Alert with Confidence {signal_data['confidence']:.1%} ---")
print(f"Regime Hurdle (forced): {signal_data['regime_threshold']:.1%}")
print(f"Global Alert Threshold: {notifier.telegram_config['alert_threshold']:.1%}")

# We'll mock send_telegram_message to just return True and print
def mock_send(msg):
    print("\n[MOCK TELEGRAM] SENDING MESSAGE:")
    print(msg)
    return True

notifier.send_telegram_message = mock_send

success = notifier.send_signal_alert(signal_data)
if success:
    print("\n✅ SUCCESS: Notification passed the logic gates.")
else:
    print("\n❌ FAILED: Notification was blocked.")
