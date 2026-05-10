import json

data = json.load(open('config/trading_whitelist.json'))
matrix = data.get('performance_matrix', {})

approved = []
for symbol, symbol_data in matrix.items():
    for k, v in symbol_data.items():
        if isinstance(v, dict):
            # Check if it's a direction (BUY/SELL)
            if k in ['BUY', 'SELL']:
                for tier, info in v.items():
                    if isinstance(info, dict) and info.get('status') == 'APPROVED':
                        approved.append(f'{symbol} {k} at {tier}% (Win Rate: {info.get("accuracy", 0):.1%}, Trades: {info.get("trades", 0)})')
            else:
                # Legacy format (k is the tier)
                if v.get('status') == 'APPROVED':
                    approved.append(f'{symbol} LEGACY_ALL_DIRS at {k}% (Win Rate: {v.get("accuracy", 0):.1%}, Trades: {v.get("trades", 0)})')

print('APPROVED MODELS:')
if approved:
    for a in sorted(set(approved)):
        print(f'- {a}')
else:
    print('None.')
