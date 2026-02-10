
import sys
import os
import yaml
import asyncio
from telegram import Bot

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def send_test_message():
    print("Loading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    telegram_config = config.get("notifications", {}).get("telegram", {})
    token = telegram_config.get("bot_token")
    chat_id = telegram_config.get("chat_id")
    
    if not token or not chat_id:
        print("Error: Telegram credentials not found in config.yaml")
        return

    print(f"Token: {token[:5]}...")
    print(f"Chat ID: {chat_id}")
    
    bot = Bot(token=token)
    message = (
        "🔔 **Apex Forex SaaS - Test Alert** 🔔\n\n"
        "If you are reading this, your **Sentinel** is ready to send automatic signals.\n"
        "Sleep well! 😴📈"
    )
    
    print("Sending message...")
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        print("✅ Message sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send message: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_message())
