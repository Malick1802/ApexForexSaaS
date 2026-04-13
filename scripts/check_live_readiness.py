import sys, yaml, json, urllib.request
sys.path.insert(0, '.')

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

tg = cfg.get('notifications', {}).get('telegram', {})
mt5c = cfg.get('mt5', {})

print('=== TELEGRAM CONFIG ===')
print('  Enabled:    ', tg.get('enabled'))
print('  Bot Token:  ', str(tg.get('bot_token', ''))[:12] + '...(hidden)')
print('  Chat ID:    ', tg.get('chat_id'))
print('  Threshold:  ', tg.get('alert_threshold'))

print('')
print('=== MT5 CONFIG ===')
print('  Enabled:    ', mt5c.get('enabled'))
print('  Risk Type:  ', mt5c.get('risk_type'))
print('  Risk Value: ', str(mt5c.get('risk_value')) + '%')
print('  Max Trades: ', mt5c.get('max_open_trades'))

print('')
print('=== SHADOW MODE ===')
execute = cfg.get('trading', {}).get('execute_trades', False)
print('  execute_trades:', execute)
if execute:
    print('  Status: LIVE MODE -- trades will execute in MT5')
else:
    print('  Status: SHADOW MODE -- trades are SIMULATED (ghost log only)')

# Test Telegram connectivity
token = tg.get('bot_token', '')
print('')
print('=== TELEGRAM CONNECTIVITY TEST ===')
try:
    url = 'https://api.telegram.org/bot' + token + '/getMe'
    resp = urllib.request.urlopen(url, timeout=8)
    data = json.loads(resp.read())
    if data.get('ok'):
        bot = data['result']
        print('  Connected! Bot:', bot.get('username'), '(' + bot.get('first_name') + ')')
    else:
        print('  ERROR: API returned not-ok:', data)
except Exception as e:
    print('  FAILED:', e)
