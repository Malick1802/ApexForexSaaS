
def analyze_gold_fleet(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    gold_fleet = []
    
    for line in lines:
        if line.startswith('|') and 'Pair' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                pair = parts[1]
                type_ = parts[2]
                target = parts[3]
                wr_str = parts[5].replace('%', '')
                vol_str = parts[6]
                
                try:
                    wr = float(wr_str)
                    vol = int(vol_str)
                    target_val = int(target.replace('%', ''))
                    
                    if target_val >= 90 and wr >= 90.0 and vol >= 800:
                        gold_fleet.append({
                            "pair": pair,
                            "type": type_,
                            "target": target,
                            "wr": wr,
                            "vol": vol
                        })
                except ValueError:
                    continue
                    
    return gold_fleet

if __name__ == "__main__":
    report_path = r'c:\Users\artem\Downloads\ApexForexSaaS\models\selective_accuracy_report.md'
    gold = analyze_gold_fleet(report_path)
    
    # Sort by WR descending, then Vol descending
    gold.sort(key=lambda x: (x['wr'], x['vol']), reverse=True)
    
    print(f"Total Gold Models (>=90% WR, >=800 Trades): {len(gold)}")
    for g in gold:
        print(f"{g['pair']} {g['type']} ({g['target']}): {g['wr']}% WR | {g['vol']} Trades")
