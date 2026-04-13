import sys
import yaml
import requests
from pathlib import Path

# Add project root so core modules are available
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def debug_telegram():
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    telegram = config.get('notifications', {}).get('telegram', {})
    token = telegram.get('bot_token')
    chat_id = telegram.get('chat_id')
    
    if not token or not chat_id:
        print("Token or Chat ID missing in config.yaml")
        return
        
    print(f"Token: {token[:5]}...{token[-5:]}")
    print(f"Chat ID: {chat_id}")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    message = "🛠️ This is a MOCK test message for GOLD from Apex SaaS."
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    print("Sending request to Telegram API...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    debug_telegram()
