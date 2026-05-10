import json

# Read raw OOS results directly from the audit
data = json.load(open('logs/fleet_oos_results_static.json'))

MIN_TRADES = 2
MIN_ACCURACY = 0.70

print("=" * 60)
print("  14-DAY OOS AUDIT RESULTS (Apr 15 - Apr 29, 2026)")
print("=" * 60)

passed = []
notable = []  # Has trades but below threshold

for symbol, directions in data.items():
    for direction, tiers in directions.items():
        best_tier = None
        best_acc = 0
        best_trades = 0
        for tier_str, stats in tiers.items():
            trades = stats.get('trades', 0)
            accuracy = stats.get('accuracy', 0.0)
            if trades >= MIN_TRADES and accuracy >= MIN_ACCURACY:
                if accuracy > best_acc or (accuracy == best_acc and trades > best_trades):
                    best_acc = accuracy
                    best_trades = trades
                    best_tier = tier_str
            elif trades > 0:
                notable.append((symbol, direction, tier_str, accuracy, trades))

        if best_tier:
            passed.append((symbol, direction, best_tier, best_acc, best_trades))

print(f"\n{'PASSED':=<60}")
print(f"{'PAIR':<12} {'DIR':<6} {'BEST TIER':<12} {'ACCURACY':<12} {'TRADES'}")
print("-" * 60)
if passed:
    for sym, d, tier, acc, t in sorted(passed, key=lambda x: -x[3]):
        print(f"{sym:<12} {d:<6} {tier+'%':<12} {acc*100:.1f}%{'':<7} {t}")
else:
    print("  No pairs met the validation criteria (>=70% accuracy, >=2 resolved trades)")

print(f"\n{'ALL SIGNALS WITH RESOLVED TRADES':=<60}")
print(f"{'PAIR':<12} {'DIR':<6} {'TIER':<8} {'ACCURACY':<12} {'TRADES'}")
print("-" * 60)
all_resolved = [(s,d,t,a,tr) for s,d,t,a,tr in notable] + [(s,d,t,a,tr) for s,d,t,a,tr in passed]
for sym, d, tier, acc, t in sorted(all_resolved, key=lambda x: (-x[3], -x[4])):
    marker = " [PASS]" if acc >= MIN_ACCURACY and t >= MIN_TRADES else ""
    print(f"{sym:<12} {d:<6} {tier+'%':<8} {acc*100:.1f}%{'':<7} {t}{marker}")

print(f"\nTotal resolved trades across all pairs: {sum(tr for _,_,_,_,tr in all_resolved)}")
