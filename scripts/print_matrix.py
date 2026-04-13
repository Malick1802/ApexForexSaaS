import json
import operator
import os

with open('config/trading_whitelist.json', 'r') as f:
    data = json.load(f)

matrix = data.get('performance_matrix', {})

markdown = ["# Current Fleet Performance Matrix", ""]

for symbol, tiers in sorted(matrix.items()):
    markdown.append(f"## {symbol}")
    markdown.append("| Tier | Status | Accuracy | Trades |")
    markdown.append("|---|---|---|---|")
    
    # Sort tiers numerically
    for tier, stats in sorted(tiers.items(), key=lambda item: int(item[0])):
        status = stats.get('status', 'BENCHED')
        acc = stats.get('accuracy', 0.0)
        trades = stats.get('trades', 0)
        
        status_icon = "🟢 APPROVED" if status == "APPROVED" else ("🟡 BENCHED" if trades > 0 else "⚪ NO DATA")
        
        markdown.append(f"| {tier}% | {status_icon} | {acc:.1%} | {trades} |")
    markdown.append("")

out_path = os.path.join(os.environ.get('APPDATA_DIR', '.'), 'brain', os.environ.get('CONVERSATION_ID', 'tmp'), 'performance_matrix.md')
# For testing locally, just print it directly. I'll read output and format artifact.
print("\n".join(markdown))
