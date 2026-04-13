import sys
import yaml
import sqlite3
from pathlib import Path

# Add project root so core modules are available
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.core.notifications import NotificationManager

def mock_gold_alert():
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    notifier = NotificationManager()
    
    # Check if telegram is enabled
    if not notifier.enabled:
        print("❌ Telegram notifications are disabled in config.yaml. Please enable them first.")
        return
        
    # Get the active Gold signal from the database
    db_path = PROJECT_ROOT / "signals.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM signals 
        WHERE symbol = 'GOLD' AND outcome = 'ACTIVE' AND signal IN ('BUY', 'SELL')
        ORDER BY timestamp DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("❌ No ACTIVE BUY/SELL signal for GOLD found in the database.")
        return
        
    signal_data = dict(row)
    
    # If the MT5 script hasn't updated suggested_lots properly in local testing, fallback to 0.01 for the notification visual
    if not signal_data.get('suggested_lots'):
        signal_data['suggested_lots'] = 0.01
        
    print(f"🚀 Sending Telegram alert for GOLD {signal_data['signal']} (Conf: {signal_data['confidence']*100:.1f}%)")
    
    # Temporarily bypass the confidence threshold for this mock notification
    notifier.telegram_config['alert_threshold'] = 0.0
    
    success = notifier.send_signal_alert(signal_data)
    
    if success:
        print("✅ Telegram alert sent successfully!")
    else:
        print("❌ Failed to send Telegram alert. Check your bot token and chat ID.")

if __name__ == "__main__":
    mock_gold_alert()
