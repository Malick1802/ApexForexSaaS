import json
from datetime import datetime, timezone

with open('config/trading_whitelist.json', 'r') as f:
    data = json.load(f)

if 'GOLD' in data['performance_matrix']:
    tier = data['performance_matrix']['GOLD']['60']
    # Revert to real recorded data: 1 SUCCESS, 2 FAILs
    # Bayesian prior: alpha=2 base + 1 win = 3, beta=2 base + 2 losses = 4
    tier['alpha'] = 3.0
    tier['beta'] = 4.0
    tier['accuracy'] = 0.0  # Bayesian mean = 3/7 = 43% < 70% threshold
    tier['trades'] = 3
    tier['status'] = 'BENCHED'
    tier['source'] = 'Real DB History (1W/2L)'
    tier['last_updated'] = datetime.now(timezone.utc).isoformat()
    print("GOLD 60% tier reverted to real recorded history: 1W / 2L => BENCHED")

with open('config/trading_whitelist.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Whitelist saved.")
