
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from models.specialist_factory import SpecialistFactory

logging.basicConfig(level=logging.INFO)

def diagnose_full_pipeline(symbol="AUDCAD"):
    factory = SpecialistFactory(
        min_win_rate=0.70,
        min_samples=1000,
        provider_name="mt5"
    )
    
    print(f"\n{'='*50}")
    print(f"DIAGNOSING FULL PIPELINE: {symbol}")
    print(f"{'='*50}")
    
    # 1. Fetch Labeled
    print("1. Fetching raw labeled data...")
    df = factory.engine.fetch_labeled(symbol, interval="1h", days=729)
    print(f"   -> Raw rows: {len(df)}")
    
    # 2. Extract Base Features
    print("2. Extracting base features...")
    features = factory.feature_engineer.extract_features(df)
    print(f"   -> Features rows: {len(features)}")
    
    # 3. Add Correlated
    print("3. Adding correlated assets...")
    correlated = factory.engine.get_correlated_assets(symbol)
    if correlated:
        corr_symbol = correlated[0]['symbol']
        print(f"   -> Correlated asset: {corr_symbol}")
        corr_df = factory.engine.fetch(corr_symbol, interval="1h", days=729)
        print(f"   -> Correlated raw rows: {len(corr_df)}")
        features = factory.feature_engineer.add_correlated_asset(features, corr_df)
        print(f"   -> Features after correlation: {len(features)}")
    
    # 4. Final dropna check
    print("4. Checking for NaNs and final row count...")
    # SpecialistFactory 149: y_binary = (y_all == target_label).astype(int)
    # The sequences method drops NaNs from 'combined'
    combined = features.copy()
    combined['label'] = 0 # Dummy
    final_clean = combined.dropna()
    print(f"   -> Rows after dropna(): {len(final_clean)}")
    
    # 5. Sequences
    print("5. Generating sequences (length=50)...")
    X, y = factory.feature_engineer.create_sequences(features, pd.Series(0, index=features.index))
    print(f"   -> Final X shape: {X.shape}")

if __name__ == "__main__":
    diagnose_full_pipeline("AUDCAD")
    diagnose_full_pipeline("EURUSD")
