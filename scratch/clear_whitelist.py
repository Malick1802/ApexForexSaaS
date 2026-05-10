import json
import os
from datetime import datetime

whitelist_path = r'c:\Users\artem\Downloads\ApexForexSaaS\config\trading_whitelist.json'

with open(whitelist_path, 'r') as f:
    data = json.load(f)

# Reset performance_matrix
new_matrix = {}
for symbol in data.get('performance_matrix', {}).keys():
    new_matrix[symbol] = {
        "BUY": {},
        "SELL": {}
    }

data['performance_matrix'] = new_matrix
data['last_updated'] = datetime.utcnow().isoformat() + "Z"

with open(whitelist_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Successfully reset performance matrix in {whitelist_path}")
