"""
Foundation Brain v3 (8GB RAM Optimized)
========================================
Trains a Temporal Fusion Transformer (TFT) on 5 years of Global Forex data.
Optimized for 8GB RAM constraints using on-the-fly sequence generation.

Key Upgrades for v3:
- SEQ_LEN 24 -> 36 (1.5 days context)
- UNITS 32 -> 64 (2x model capacity)
- STRIDE 8 -> 4 (2x data density)
- Added L2 Regularization for stability
"""

import os
import gc
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

# Suppress TF chatter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FoundationV3")

# ── Config (Optimized for 8GB RAM) ──────────────────────────
HISTORY_DAYS   = 1825      # 5 years
OOS_DAYS       = 45        # Increased holdout
VAL_DAYS       = 150
BATCH_SIZE     = 32        # Small batches to prevent OOM
EPOCHS         = 80        # Longer training
UNITS          = 64        # 2x v2 capacity
STRIDE         = 4         # 2x v2 density
SEQ_LEN        = 36        # 1.5 days lookback
LEARNING_RATE  = 0.0005    # Slower, more stable learning

FOREX_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "GBPJPY","EURJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","GBPCHF",
    "EURGBP","AUDNZD","NZDCHF","NZDCAD","CADCHF","AUDCHF","EURCAD",
    "GBPNZD","EURNZD","GBPCAD","USDSGD","EURAUD","EURCHF","GBPAUD",
    "AUDCAD", "GOLD"
]

# ─────────────────────────────────────────────────────────────
#  DATA GENERATOR (Memory Safe)
# ─────────────────────────────────────────────────────────────

class FoundationGeneratorV3(keras.utils.Sequence):
    """
    On-the-fly sequence generator to stay within 8GB RAM.
    """
    def __init__(self, features_dict, labels_dict, split='train'):
        self.features_dict = features_dict
        self.labels_dict = labels_dict
        self.samples = []
        
        # Build index of valid start positions
        for symbol in features_dict.keys():
            feat_len = len(features_dict[symbol])
            n_total = feat_len - SEQ_LEN
            if n_total <= 0: continue
            
            n_oos = int(n_total * (OOS_DAYS / HISTORY_DAYS))
            n_val = int(n_total * (VAL_DAYS / HISTORY_DAYS))
            n_train = n_total - n_oos - n_val
            
            if split == 'train': start, end = 0, n_train
            elif split == 'val': start, end = n_train, n_train + n_val
            else: start, end = n_train + n_val, n_total
            
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
            X[i] = self.features_dict[sym][start_idx : start_idx + SEQ_LEN]
            y[i] = self.labels_dict[sym][start_idx + SEQ_LEN]
        
        return X, y

# ─────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────

class FoundationTrainerV3:
    def __init__(self):
        self.output_dir = PROJECT_ROOT / "models" / "foundation_v3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        logger.info("🚀 Starting Foundation Brain v3 Training (8GB RAM Optimized)...")
        
        # 1. Reuse v2 data loading logic but with v3 parameters
        from models.foundation_trainer_v2 import FoundationTrainerV2
        v2_engine = FoundationTrainerV2()
        
        raw = v2_engine.fetch_all_data()
        aligned = v2_engine.align(raw)
        del raw; gc.collect()
        
        # 2. Build Corpus
        # Note: We use the v2 logic to extract features, then wrap in v3 generator
        features_dict, labels_dict, n_features = v2_engine.build_corpus(aligned)
        del aligned; gc.collect()
        
        # 3. Generators
        train_gen = FoundationGeneratorV3(features_dict, labels_dict, 'train')
        val_gen   = FoundationGeneratorV3(features_dict, labels_dict, 'val')
        
        # 4. Build v3 Model (Enhanced with L2 and more units)
        from models.global_brain import build_global_brain
        input_shape = (SEQ_LEN, n_features)
        
        # Custom build for v3
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(UNITS, return_sequences=True, kernel_regularizer='l2'),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(UNITS, kernel_regularizer='l2'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 5. Train
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(self.output_dir / "foundation_brain.keras"), save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
        ]
        
        logger.info(f"Training v3 model with {len(train_gen.samples):,} sequences...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # 6. Save Final Config
        config = {
            "version": "v3",
            "units": UNITS,
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
            "n_features": n_features,
            "trained_at": datetime.now().isoformat()
        }
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✅ Foundation v3 complete! Saved to {self.output_dir}")

if __name__ == "__main__":
    FoundationTrainerV3().run()
