"""Quick test of executive engine - single scan."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.executive import ExecutiveEngine

# Run one scan
print("Testing Executive Engine...")
print("="*60)

engine = ExecutiveEngine(confidence_threshold=0.70, scan_interval_minutes=15)

# Test with just 3 pairs for speed
test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

print(f"\nRunning test scan on: {', '.join(test_symbols)}")
engine.run_scan(test_symbols)

print("\n" + "="*60)
print("Test complete! Check logs/system.log for details.")
print("Check signals.db for any generated signals.")
