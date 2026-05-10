import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path

# Force UTF-8 output to avoid Windows cp1252 codec crashes
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras
from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from data_pipeline.labeling import triple_barrier_label
from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork

logging.basicConfig(level=logging.INFO, format='%(message)s')

OOS_START_DATE = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'GOLD', 'USDCAD']

FEATURE_COLS = [
    'open_norm', 'high_norm', 'low_norm', 'hl_range', 'oc_range',
    'close_ret_1', 'close_ret_5', 'close_ret_10', 'rsi', 'atr_norm',
    'bb_position', 'bb_width_norm', 'macd_norm', 'macd_signal_norm',
    'macd_hist_norm', 'volume_rel', 'volume_ret', 'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos', 'USD_strength', 'EUR_strength', 'GBP_strength',
    'JPY_strength', 'AUD_strength', 'CAD_strength', 'CHF_strength',
    'NZD_strength', 'dxy_proxy', 'dxy_ret', 'gold_ret', 'vix_proxy',
    'yield_curve_slope'
]

def load_foundation():
    brain_path  = PROJECT_ROOT / "models" / "foundation" / "foundation_brain.keras"
    scaler_path = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
    if not brain_path.exists(): return None, None
    model = keras.models.load_model(
        str(brain_path),
        custom_objects={
            'VariableSelectionNetwork': VariableSelectionNetwork,
            'GatedResidualNetwork':     GatedResidualNetwork
        }
    )
    scaler = joblib.load(str(scaler_path))
    return model, scaler

def align_features(features, df):
    # Just simple renaming
    renames = {
        'atr': 'atr_norm', 'bb_width': 'bb_width_norm',
        'macd': 'macd_norm', 'macd_signal': 'macd_signal_norm',
        'macd_hist': 'macd_hist_norm', 'volume_norm': 'volume_rel'
    }
    for src, dst in renames.items():
        if src in features.columns:
            features = features.rename(columns={src: dst})
            
    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0.0
            
    return features[FEATURE_COLS]

def evaluate():
    model, scaler = load_foundation()
    if model is None:
        print("No foundation model found.")
        return
        
    engine = DataEngine()
    fe = FeatureEngineer()
    gfe = GlobalFeatureEngineer()
    
    # Pre-fetch contexts
    context_data = {}
    for s in SYMBOLS:
        df = engine.fetch(s, interval="1h", days=60)
        if not df.empty: context_data[s] = df
        
    results = {}
    
    for symbol in SYMBOLS:
        print(f"\n--- Testing Global Brain on {symbol} ---")
        df_labeled = engine.fetch_labeled(symbol, interval="1h", days=60)
        if df_labeled.empty: continue
        
        # Cut to OOS window plus history
        oos_start = pd.to_datetime(OOS_START_DATE, utc=True)
        # We need 60 bars before OOS start for the sequence
        start_idx = df_labeled.index.searchsorted(oos_start)
        if start_idx == 0: continue
        
        base_features = fe.extract_features(df_labeled)
        features = gfe.add_global_features(symbol, base_features, context_data)
        aligned = align_features(features, df_labeled)
        
        # Scale
        scaled = scaler.transform(aligned)
        
        y = df_labeled['label'].astype(int).values
        
        X, y_seq = fe.create_sequences(pd.DataFrame(scaled, columns=aligned.columns, index=aligned.index), y, sequence_length=60)
        
        # Align dates with sequences
        valid_dates = df_labeled.index[-len(X):]
        oos_mask = valid_dates >= oos_start
        
        X_oos = X[oos_mask]
        y_oos = y_seq[oos_mask]
        
        if len(X_oos) == 0: continue
        
        preds = model.predict(X_oos, verbose=0)
        
        buy_probs = preds[:, 1]
        sell_probs = preds[:, 2]
        
        for threshold in [0.6, 0.7]:
            buy_signals = (buy_probs >= threshold)
            sell_signals = (sell_probs >= threshold)
            
            total_trades = buy_signals.sum() + sell_signals.sum()
            if total_trades == 0: continue
            
            buy_wins = np.sum((buy_probs >= threshold) & (y_oos == 1))
            sell_wins = np.sum((sell_probs >= threshold) & (y_oos == 2))
            
            acc = (buy_wins + sell_wins) / total_trades
            print(f"[{symbol}] @ {threshold*100:.0f}% Conviction -> Trades: {total_trades}, Win Rate: {acc*100:.1f}%")

if __name__ == "__main__":
    evaluate()
