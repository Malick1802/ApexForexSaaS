import json
import os

with open('config/trading_whitelist.json', 'r') as f:
    data = json.load(f)

gold = data['performance_matrix'].get('GOLD', {})
print("GOLD Performance Matrix:")
print(json.dumps(gold, indent=2))
