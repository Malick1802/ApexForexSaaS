import json

data = json.load(open('logs/clean_wfa_results.json'))

print(f"\n{'='*60}")
print(f"  CLEAN WFA — {data['symbol']}")
print(f"  Average In-Sample Accuracy: {data['avg_is_accuracy']:.1%}")
print(f"{'='*60}\n")

for w in data['windows']:
    print(f"  Window {w['window']} | IS={w['is_accuracy']:.1%} | Train={w['train_samples']:,} | Test={w['test_samples']:,}")
    for t, v in w['thresholds'].items():
        print(f"    @{float(t):.0%} → OOS={v['oos_accuracy']:.1%}  WFE={v['wfe']:.2f}  Trades={v['trades']}")
    print()

print(f"\n{'─'*60}")
print("  AGGREGATE SUMMARY")
print(f"{'─'*60}")
for thresh, v in data['summary'].items():
    print(f"  @{float(thresh):.0%} | OOS={v['avg_oos_accuracy']:.1%} | WFE={v['avg_wfe']:.2f} | Trades={v['total_trades']:,} | {v['status']}")
