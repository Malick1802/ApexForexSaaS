
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.notifications import NotificationManager

def test_alert():
    print("Initializing NotificationManager...")
    nm = NotificationManager()
    
    if not nm.enabled:
        print("❌ Telegram is NOT enabled in config.yaml")
        return

    print(f"Bot Token: {nm.bot_token[:5]}...")
    print(f"Chat ID: {nm.chat_id}")
    
    message = "🔔 *Apex Forex SaaS - Restoration Test* 🔔\n\nTelegram alerts have been successfully restored for Python 3.13! 🚀"
    
    print("Sending test message...")
    success = nm.send_telegram_message(message)
    
    if success:
        print("✅ Message sent successfully!")
    else:
        print("❌ Failed to send message. Check logs/system_v2.log for details.")

if __name__ == "__main__":
    test_alert()
