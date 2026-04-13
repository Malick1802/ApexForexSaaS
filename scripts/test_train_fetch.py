
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from models.specialist_factory import SpecialistFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFetch")

def test_fetch(symbol="AUDCAD"):
    factory = SpecialistFactory(
        min_win_rate=0.70,
        min_samples=1000,
        provider_name="mt5"
    )
    
    print(f"\nTesting fetch_labeled for {symbol}...")
    try:
        df = factory.engine.fetch_labeled(symbol, interval="1h", days=729)
        if df is not None:
            print(f"SUCCESS: Fetched {len(df)} rows")
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                print(f"Labels:\n{label_counts}")
            else:
                print("WARNING: 'label' column missing!")
        else:
            print("FAILED: fetch_labeled returned None")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_fetch("AUDCAD")
    test_fetch("EURUSD")
