import json
from pathlib import Path

def generate_full_report():
    p = Path('config/trading_whitelist.json')
    if not p.exists():
        print("Error: Whitelist not found.")
        return
        
    data = json.load(open(p))
    matrix = data.get('performance_matrix', {})
    
    approved = []
    benched = []
    
    for sym, dirs in matrix.items():
        for d, tiers in dirs.items():
            if isinstance(tiers, dict):
                for t, stats in tiers.items():
                    if isinstance(stats, dict) and 'status' in stats:
                        entry = {
                            'sym': sym,
                            'dir': d if d in ['BUY', 'SELL'] else 'ALL',
                            'tier': t,
                            'win': stats.get('accuracy', 0),
                            'trades': stats.get('trades', 0),
                            'status': stats.get('status')
                        }
                        if entry['status'] == 'APPROVED':
                            approved.append(entry)
                        elif entry['status'] == 'BENCHED' and entry['trades'] > 0:
                            benched.append(entry)

    # Sort and Print
    print("--- APPROVED (CERTIFIED & TRADING) ---")
    if not approved:
        print("No approved pairs found.")
    else:
        approved.sort(key=lambda x: (x['sym'], -int(x['tier'])))
        for e in approved:
            print(f"[{e['sym']}] {e['dir']} @ {e['tier']}% Tier | Win Rate: {e['win']:.1%} | Trades: {e['trades']}")

    print("\n--- BENCHED (HISTORY ACCUMULATING) ---")
    if not benched:
        print("No active benched pairs with history found.")
    else:
        benched.sort(key=lambda x: (x['sym'], -int(x['tier'])))
        for e in benched:
            print(f"[{e['sym']}] {e['dir']} @ {e['tier']}% Tier | Win Rate: {e['win']:.1%} | Trades: {e['trades']}")

if __name__ == "__main__":
    generate_full_report()
