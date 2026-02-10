"""Debug executive engine crash."""
import sys
import logging
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.executive import ExecutiveEngine

logging.basicConfig(level=logging.INFO)

print("DEBUG: Initializing Executive Engine")
engine = ExecutiveEngine(confidence_threshold=0.60)

symbol = "EURUSD"
print(f"DEBUG: Analyzing {symbol}")

# Verify config loading
correlated = engine._get_correlated_assets(symbol)
print(f"DEBUG: Correlated assets for {symbol}: {correlated}")

try:
    print("DEBUG: Calling analyze_symbol...")
    result = engine.analyze_symbol(symbol)
    print(f"DEBUG: Analysis Result: {result}")
    
except Exception as e:
    print("\nCRASH DETECTED:")
    print("-" * 30)
    traceback.print_exc()
    print("-" * 30)

except Exception as e:
    print("\nCRASH DETECTED:")
    print("-" * 30)
    traceback.print_exc()
    print("-" * 30)
