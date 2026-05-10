import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from data_pipeline.labeling import triple_barrier_label
from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BiasCheck")

def check_bias():
    data_engine = DataEngine()
    feature_engineer = FeatureEngineer()
    global_engineer = GlobalFeatureEngineer()
    
    # 1. Load Foundation Brain
    brain_path = PROJECT_ROOT / "models" / "foundation" / "foundation_brain.keras"
    scaler_path = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
    
    model = keras.models.load_model(
        str(brain_path),
        custom_objects={
            'VariableSelectionNetwork': VariableSelectionNetwork,
            'GatedResidualNetwork': GatedResidualNetwork
        }
    )
    scaler = joblib.load(str(scaler_path))
    mean = scaler.mean_
    scale = scaler.scale_
    
    # 2. Setup Symbols
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "GOLD"]
    
    # 3. Global Data
    global_data = {}
    g7_pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "GOLD", "^TNX", "^IRX"]
    for g in g7_pairs:
        try:
            gdf = data_engine.fetch(g, interval="1h", days=20)
            if gdf is not None: global_data[g] = gdf
        except: pass

    # 4. Feature Alignment Logic (Exactly as in inference.py)
    f_cols = [
        'open_norm', 'high_norm', 'low_norm', 'hl_range', 'oc_range',
        'close_ret_1', 'close_ret_5', 'close_ret_10', 'rsi', 'atr_norm',
        'bb_position', 'bb_width_norm', 'macd_norm', 'macd_signal_norm',
        'macd_hist_norm', 'volume_rel', 'volume_ret', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'USD_strength', 'EUR_strength', 'GBP_strength',
        'JPY_strength', 'AUD_strength', 'CAD_strength', 'CHF_strength',
        'NZD_strength', 'dxy_proxy', 'dxy_ret', 'gold_ret', 'vix_proxy',
        'yield_curve_slope'
    ]
    mapping = {
        'atr': 'atr_norm', 'bb_width': 'bb_width_norm',
        'macd': 'macd_norm', 'macd_signal': 'macd_signal_norm',
        'macd_hist': 'macd_hist_norm', 'volume_norm': 'volume_rel'
    }

    for symbol in symbols:
        print(f"\n--- Checking {symbol} ---")
        df = data_engine.fetch(symbol, interval="1h", days=20)
        if df is None: continue
        
        base_features = feature_engineer.extract_features(df)
        features = global_engineer.add_global_features(symbol, base_features, global_data)
        
        for src, dst in mapping.items():
            if src in features.columns: features[dst] = features[src]
            
        for i in [1, 5, 10]:
            features[f'close_ret_{i}'] = df['close'].pct_change(i).fillna(0)
            
        ts = features.index
        features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24.0)
        features['dow_sin'] = np.sin(2 * np.pi * ts.weekday / 7.0)
        features['dow_cos'] = np.cos(2 * np.pi * ts.weekday / 7.0)
        
        if 'volume_norm' in features.columns:
            features['volume_rel'] = features['volume_norm']
        if 'volume_ret' not in features.columns:
            if 'volume_rel' in features.columns:
                features['volume_ret'] = features['volume_rel'].pct_change().fillna(0)
            else:
                features['volume_ret'] = 0.0
            
        final_f = []
        for c in f_cols:
            if c not in features.columns: features[c] = 0.0
            final_f.append(c)
        features = features[final_f]
        
        # Sequence Creation
        X, _ = feature_engineer.create_sequences(features, pd.Series(0, index=features.index), sequence_length=60)
        if len(X) == 0: continue
        
        X_last = X[-1:].reshape(1, 60, 34)
        X_scaled = (X_last - mean) / scale
        
        preds = model.predict(X_scaled, verbose=0)[0]
        print(f"\nPREDICTIONS -> WAIT: {preds[0]:.4f} | BUY: {preds[1]:.4f} | SELL: {preds[2]:.4f}")
        
        # LABEL CHECKING
        print("\nLABEL CHECK (Last 5 bars):")
        df_labeled = triple_barrier_label(df, symbol=symbol)
        last_labels = df_labeled['label'].tail(5)
        print(last_labels)
        print(f"Distribution: {df_labeled['label'].value_counts().to_dict()}")

        print("\nTop 10 Active Features (Z-Score):")
        z_scores = X_scaled[0, -1, :]
        feat_z = sorted(zip(f_cols, z_scores), key=lambda x: abs(x[1]), reverse=True)
        for name, val in feat_z[:10]:
            print(f"  {name}: {val:.2f}")

if __name__ == "__main__":
    check_bias()
