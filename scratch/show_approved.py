import json

data = json.load(open('config/trading_whitelist.json'))
approved = []
benched = []

for symbol, sdata in data.items():
    if symbol == 'last_updated': continue
    for direction in ['BUY', 'SELL']:
        if direction in sdata:
            for tier, metrics in sdata[direction].items():
                accuracy = metrics.get('accuracy', 0) * 100
                trades = metrics.get('trades', 0)
                status = metrics.get('status')
                
                if status == 'APPROVED':
                    approved.append(f'- **{symbol} {direction}** (Tier {tier}%): Accuracy {accuracy:.1f}% over {trades} trades')
                elif trades > 0:
                    benched.append((accuracy, trades, f'- **{symbol} {direction}** (Tier {tier}%): Accuracy {accuracy:.1f}% over {trades} trades - {status}'))

for line in approved:
    print(line)

if not approved:
    print('No pairs passed the strict APPROVED validation criteria (typically >70% accuracy and >2 trades in the 14-day window).')

print("\nHere are the top performing BENCHED pairs that generated trades but didn't cross the minimum threshold:")
benched.sort(key=lambda x: (x[0], x[1]), reverse=True)
for i, item in enumerate(benched[:10]):
    print(item[2])
