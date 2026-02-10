
import re

def parse_report(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    low_volume_pairs = []
    
    for line in lines:
        if line.startswith('|') and 'Pair' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                pair = parts[1]
                target = parts[3]
                volume_str = parts[6]
                
                try:
                    volume = int(volume_str)
                    if volume < 100 and target in ['90%', '95%']:
                        low_volume_pairs.append((pair, target, volume))
                except ValueError:
                    continue
                    
    return low_volume_pairs

if __name__ == "__main__":
    report_path = r'c:\Users\artem\Downloads\ApexForexSaaS\models\selective_accuracy_report.md'
    pairs = parse_report(report_path)
    
    # Sort by pair name
    pairs.sort()
    
    unique_pairs = sorted(list(set([p[0] for p in pairs])))
    print(f"Found {len(pairs)} model configurations with < 100 trades across {len(unique_pairs)} unique pairs.")
    print("\nUnique pairs to retrain:")
    print(", ".join(unique_pairs))
    
    print("\nFull List:")
    for p, t, v in pairs:
        print(f"{p} ({t}) - Volume: {v}")
