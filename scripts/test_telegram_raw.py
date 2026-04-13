import yaml

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

tg = cfg.get('notifications', {}).get('telegram', {})
print(f"Token: {tg.get('bot_token')}")
print(f"Chat: {tg.get('chat_id')}")

# Test sending a mock message
import requests
token = tg.get('bot_token')
chat_id = tg.get('chat_id')
url = f"https://api.telegram.org/bot{token}/sendMessage"
try:
    r = requests.post(url, json={"chat_id": chat_id, "text": "Test message from ApexForex system diagnostic."})
    print(r.json())
except Exception as e:
    print(e)
