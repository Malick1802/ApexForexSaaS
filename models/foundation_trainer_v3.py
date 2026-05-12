"""
Foundation Brain v3 "The Fortress"
===================================
A rigorous, zero-leakage Global Brain trainer.
Uses Rolling Z-Score normalization and strict temporal isolation.

Optimized for 8GB RAM.
Macro Features: SP500, GOLD, OIL, NASDAQ, US10Y, VIX
"""

import os
import gc
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Suppress TF chatter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FortressV3")

# ── Config (Rigorous Isolation) ─────────────────────────────
HISTORY_DAYS   = 1825
OOS_DAYS       = 60        # Strict 2-month final test
VAL_DAYS       = 180       # 6-month validation
DEAD_ZONE_DAYS = 2         # Absolute isolation buffer
SEQ_LEN        = 36        # 1.5 days context
UNITS          = 64
STRIDE         = 4
BATCH_SIZE     = 32
LEARNING_RATE  = 0.0003    # Lower for precision
EPOCHS         = 100

FOREX_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "GBPJPY","EURJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","GBPCHF",
    "EURGBP","AUDNZD","NZDCHF","NZDCAD","CADCHF","AUDCHF","EURCAD",
    "GBPNZD","EURNZD","GBPCAD","EURAUD","EURCHF","GBPAUD","AUDCAD"
]

MACRO_SYMBOLS = ["S&P500", "GOLD", "CrudeOIL", "NASDAQ", "US10Y", "VIX"]

# ─────────────────────────────────────────────────────────────
#  MEMORY-SAFE DATA ENGINE
# ─────────────────────────────────────────────────────────────

class FortressGenerator(keras.utils.Sequence):
    def __init__(self, features_dict, labels_dict, split='train'):
        self.features_dict = features_dict
        self.labels_dict = labels_dict
        self.samples = []
        
        for symbol in features_dict.keys():
            feat_len = len(features_dict[symbol])
            n_total = feat_len - SEQ_LEN
            
            # Strict split indices
            oos_start = n_total - int(n_total * (OOS_DAYS / HISTORY_DAYS))
            val_start = oos_start - int(n_total * (VAL_DAYS / HISTORY_DAYS))
            train_end = val_start - int(n_total * (DEAD_ZONE_DAYS / HISTORY_DAYS))
            
            if split == 'train': start, end = 0, train_end
            elif split == 'val': start, end = val_start, oos_start
            else: start, end = oos_start, n_total
            
            for i in range(start, end, STRIDE):
                self.samples.append((symbol, i))
        
        if split == 'train': np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / BATCH_SIZE))

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        n_feat = next(iter(self.features_dict.values())).shape[1]
        
        X = np.empty((len(batch_samples), SEQ_LEN, n_feat), dtype=np.float32)
        y = np.empty((len(batch_samples),), dtype=np.int32)
        
        for i, (sym, start_idx) in enumerate(batch_samples):
            # Rolling Z-Score (Normalization happens here per sample - zero leakage!)
            raw_seq = self.features_dict[sym][start_idx : start_idx + SEQ_LEN]
            mean = np.mean(raw_seq, axis=0)
            std = np.std(raw_seq, axis=0) + 1e-8
            X[i] = (raw_seq - mean) / std
            
            y[i] = self.labels_dict[sym][start_idx + SEQ_LEN]
        
        return X, y

# ─────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────

class FortressTrainerV3:
    def __init__(self):
        self.output_dir = PROJECT_ROOT / "models" / "foundation_v3_fortress"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_data(self):
        from data_pipeline.engine import DataEngine
        engine = DataEngine()
        
        data = {}
        all_symbols = FOREX_PAIRS + MACRO_SYMBOLS
        
        logger.info(f"Fetching {len(all_symbols)} symbols (5 Years)...")
        for sym in all_symbols:
            try:
                df = engine.fetch(sym, "1h", days=HISTORY_DAYS)
                if df is not None and not df.empty:
                    data[sym] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {sym}: {e}")
        return data

    def engineer_features(self, data_dict):
        logger.info("Engineering Rigorous Features + Macro Alignment...")
        processed = {}
        
        # 1. Prepare Macros
        macro_dfs = {s: data_dict[s] for s in MACRO_SYMBOLS if s in data_dict}
        
        for sym in FOREX_PAIRS:
            if sym not in data_dict: continue
            df = data_dict[sym].copy()
            
            # Technicals (Relative only)
            df['returns'] = df['close'].pct_change()
            df['range'] = (df['high'] - df['low']) / df['close']
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # Alignment with Macros
            for m_sym, m_df in macro_dfs.items():
                m_close = m_df['close'].reindex(df.index, method='ffill')
                df[f'macro_{m_sym}_ret'] = m_close.pct_change()
            
            # Labeling (The Race: 1:1.5 RR)
            df['label'] = self.generate_labels(df)
            
            df = df.dropna()
            processed[sym] = df
            
        return processed

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def generate_labels(self, df):
        # 1:1.5 RR Race (Purged)
        # 1 = BUY, 2 = SELL, 0 = WAIT
        close = df['close'].values
        labels = np.zeros(len(df))
        
        for i in range(len(df) - 100):
            p0 = close[i]
            tp = p0 * 0.003
            sl = tp / 1.5
            
            for j in range(i + 1, min(i + 100, len(df))):
                diff = close[j] - p0
                if diff >= tp: 
                    labels[i] = 1 # BUY
                    break
                if diff <= -sl:
                    labels[i] = 2 # SELL
                    break
        return labels

    def run(self):
        raw = self.fetch_data()
        processed = self.engineer_features(raw)
        del raw; gc.collect()
        
        features_dict = {}
        labels_dict = {}
        
        for sym, df in processed.items():
            feat_cols = [c for c in df.columns if c not in ['label', 'open', 'high', 'low', 'close', 'tick_volume']]
            features_dict[sym] = df[feat_cols].values.astype(np.float32)
            labels_dict[sym] = df['label'].values.astype(np.int32)
            
        n_features = features_dict[next(iter(features_dict))].shape[1]
        
        # 4. Sanity Check: Label Distribution
        unique, counts = np.unique(np.concatenate(list(labels_dict.values())), return_counts=True)
        dist = dict(zip(unique, counts))
        total = sum(counts)
        logger.info(f"📊 Label Distribution: { {int(k): f'{(v/total)*100:.1f}%' for k, v in dist.items()} }")
        
        baseline = max(counts) / total
        logger.info(f"⚖️ Baseline Accuracy (Random Guessing Majority): {baseline*100:.1f}%")
        
        train_gen = FortressGenerator(features_dict, labels_dict, 'train')
        val_gen   = FortressGenerator(features_dict, labels_dict, 'val')
        
        # 5. Build Fortress Model
        model = keras.Sequential([
            keras.layers.Input(shape=(SEQ_LEN, n_features)),
            keras.layers.LSTM(UNITS, return_sequences=True, kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.LSTM(UNITS, kernel_regularizer='l2'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(self.output_dir / "foundation_brain.keras"), save_best_only=True)
        ]
        
        logger.info(f"Training Fortress v3 with {n_features} features (including macros)...")
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
        
        # Save Metadata
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump({"version": "v3_fortress", "features": n_features, "trained_at": datetime.now().isoformat()}, f)

if __name__ == "__main__":
    FortressTrainerV3().run()
