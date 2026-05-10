import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from models.global_brain import build_global_brain
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import gc

logger = logging.getLogger(__name__)

class GlobalBrainTrainer:
    """
    Orchestrates the training of the Global Market Brain across all pairs.
    """
    def __init__(
        self,
        symbols: List[str] = None,
        history_days: int = 730, # 2 years for global training
        batch_size: int = 128,
        epochs: int = 50,
        units: int = 64
    ):
        self.engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        self.global_engineer = GlobalFeatureEngineer()
        
        self.symbols = symbols or self.engine.get_all_pairs()
        if "GOLD" not in self.symbols:
            self.symbols.append("GOLD")
        if "^TNX" not in self.symbols:
            self.symbols.append("^TNX")
            
        self.history_days = history_days
        self.batch_size = batch_size
        self.epochs = epochs
        self.units = units
        
        self.model_dir = Path("models/foundation")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path("/tmp/global_training")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def prepare_global_dataset(self):
        """
        Fetches, aligns, and merges data from all symbols into DISK-BACKED training pools.
        
        CRITICAL FIX: Incremental Disk Dumping
        -----------------------------------------------
        To avoid ArrayMemoryError for 5-year historical training, we:
        1. Alight all data to a single unified timeline.
        2. Process each symbol one at a time and save to .npy chunks.
        3. Fit the scaler incrementally via partial_fit.
        """
        logger.info(f"Preparing global dataset for {len(self.symbols)} symbols (1825-day horizon)...")
        
        # Clear old data
        for f in self.data_dir.glob("*.npy"): 
            try: os.remove(f)
            except: pass
        
        # 1. Fetch raw data for all context assets (Gold, Yields, etc.)
        raw_data = {}
        for symbol in self.symbols:
            try:
                # Fetch more than history_days to allow for labeling window
                df = self.engine.fetch(symbol, interval="1h", days=self.history_days + 30)
                if not df.empty:
                    raw_data[symbol] = df
                    logger.info(f"Loaded {symbol}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        # 2. Universal Alignment (The Master Timeline)
        common_index = None
        for symbol, df in raw_data.items():
            if common_index is None: common_index = df.index
            else: common_index = common_index.intersection(df.index)
        
        logger.info(f"Synchronized Timeline: {len(common_index)} overlapping hours")
        aligned_data = {s: df.reindex(common_index).ffill().bfill() for s, df in raw_data.items()}
        
        # 3. Incremental Scaler Fitting & Disk Dumping
        scaler = StandardScaler()
        feature_names = []
        train_stats = {"samples": 0}
        val_stats = {"samples": 0}
        
        for symbol in self.symbols:
            if symbol in ["GOLD", "^TNX"]: continue
            
            try:
                logger.info(f"Processing {symbol} into disk chunks...")
                
                df_labeled = self.engine.fetch_labeled(symbol, interval="1h", days=self.history_days + 30)
                df_labeled = df_labeled.reindex(common_index).ffill().bfill()
                
                base_features = self.feature_engineer.extract_features(df_labeled)
                features = self.global_engineer.add_global_features(symbol, base_features, aligned_data)
                
                y = df_labeled['label'].astype(int).values
                X, y_seq = self.feature_engineer.create_sequences(features, y, sequence_length=60)
                
                if len(X) == 0: continue
                
                # Split at 90%
                split = int(len(X) * 0.90)
                X_train, y_train = X[:split], y_seq[:split]
                X_val, y_val = X[split:], y_seq[split:]
                
                # Partial Fit Scaler on training portion
                scaler.partial_fit(X_train.reshape(-1, X_train.shape[2]))
                
                # Save to disk
                np.save(self.data_dir / f"{symbol}_train_X.npy", X_train.astype(np.float32))
                np.save(self.data_dir / f"{symbol}_train_y.npy", y_train.astype(np.float32))
                np.save(self.data_dir / f"{symbol}_val_X.npy", X_val.astype(np.float32))
                np.save(self.data_dir / f"{symbol}_val_y.npy", y_val.astype(np.float32))
                
                train_stats["samples"] += len(X_train)
                val_stats["samples"] += len(X_val)
                feature_names = list(features.columns)
                
                # Critical Memory Management
                del X, y_seq, X_train, y_train, X_val, y_val, features, base_features
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")
                continue

        # Save Scaler and Feature Metadata
        joblib.dump(scaler, self.model_dir / "scaler.joblib")
        logger.info(f"Dataset prepared and Scaler saved. Total samples: {train_stats['samples'] + val_stats['samples']}")
        return feature_names, train_stats, val_stats


    def train(self):
        """Train the Foundation Model with Disk-Backed Streaming."""
        # Process and dump data to disk
        feature_names, train_stats, val_stats = self.prepare_global_dataset()
        
        # Load scaler to pre-scale the data (actually we should have done it during dump)
        # To avoid re-dumping, we'll implement a scaling generator
        scaler = joblib.load(self.model_dir / "scaler.joblib")
        mean = scaler.mean_.astype(np.float32)
        scale = scaler.scale_.astype(np.float32)
        
        num_feats = len(feature_names)
        seq_len = 60 # Default
        
        logger.info("Building Disk-Backed Generator for deep 5-year training...")
        
        class DiskChunkGenerator(tf.keras.utils.Sequence):
            def __init__(self, data_dir, symbols, mode='train', batch_size=256, mean=None, scale=None):
                self.data_dir = data_dir
                self.symbols = symbols
                self.mode = mode
                self.batch_size = batch_size
                self.mean = mean
                self.scale = scale
                
                self.files_X = [data_dir / f"{s}_{mode}_X.npy" for s in symbols if (data_dir / f"{s}_{mode}_X.npy").exists()]
                self.files_y = [data_dir / f"{s}_{mode}_y.npy" for s in symbols if (data_dir / f"{s}_{mode}_y.npy").exists()]
                
                # Pre-calculate file boundaries for absolute indexing
                self.file_lengths = [len(np.load(f, mmap_mode='r')) for f in self.files_X]
                self.total_samples = sum(self.file_lengths)
                
                # Pre-map indices: List of (file_idx, start_in_file)
                self.indices = []
                for f_idx, length in enumerate(self.file_lengths):
                    for start in range(0, length, batch_size):
                        self.indices.append((f_idx, start))
                
                np.random.shuffle(self.indices)

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                f_idx, start = self.indices[idx]
                end = min(start + self.batch_size, self.file_lengths[f_idx])
                
                # Memory-mapped load (extremely fast, zero copy until slice)
                X = np.load(self.files_X[f_idx], mmap_mode='r')[start:end]
                y = np.load(self.files_y[f_idx], mmap_mode='r')[start:end]
                
                # Apply Scaling on the fly
                X_flat = X.reshape(-1, X.shape[2]).copy()
                X_flat -= self.mean
                X_flat /= self.scale
                
                return X_flat.reshape(X.shape), y
                
            def on_epoch_end(self):
                np.random.shuffle(self.indices)

        train_gen = DiskChunkGenerator(self.data_dir, self.symbols, 'train', self.batch_size, mean, scale)
        val_gen = DiskChunkGenerator(self.data_dir, self.symbols, 'val', self.batch_size, mean, scale)
        
        input_shape = (seq_len, num_feats)
        model = build_global_brain(input_shape, units=self.units, num_classes=3)
        
        self.checkpoint_path = self.model_dir / "checkpoint.weights.h5"
        
        # Resume Logic
        if self.checkpoint_path.exists():
            logger.info(f"🔄 Resuming from checkpoint: {self.checkpoint_path}")
            try: model.load_weights(str(self.checkpoint_path))
            except Exception as e: logger.warning(f"Failed to load weights: {e}")

        class LoggingCallback(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if batch % 100 == 0:
                    logger.info(f"Step {batch}: accuracy={logs.get('accuracy', 0):.4f}, loss={logs.get('loss', 0):.4f}")
            def on_epoch_end(self, epoch, logs=None):
                val_acc = logs.get('val_accuracy', float('nan'))
                val_loss = logs.get('val_loss', float('nan'))
                logger.info(f"Epoch {epoch + 1} finished: accuracy={val_acc:.4f}, loss={val_loss:.4f}")

        # Compute class weights (We'll use a representative sample or the stats)
        # For speed, we'll use a fixed weight or compute from a small subset
        class_weight = {0: 0.5, 1: 2.0, 2: 2.0} # Standard imbalance compensation
        logger.info(f"Applying institutional class weights: {class_weight}")
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            verbose=0,
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.checkpoint_path),
                    save_weights_only=True,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                LoggingCallback()
            ]
        )
        
        # Save results final
        model.save(self.model_dir / "foundation_brain.keras")
        
        # Clean up
        if self.checkpoint_path.exists(): os.remove(self.checkpoint_path)
        for f in self.data_dir.glob("*.npy"): 
            try: os.remove(f)
            except: pass
        
        config = {
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat(),
            "symbols_count": len(self.symbols),
            "total_samples": train_stats["samples"] + val_stats["samples"],
            "final_accuracy": float(history.history['val_accuracy'][-1])
        }
        
        with open(self.model_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✅ Enhanced Global Foundation Model Saved to {self.model_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = GlobalBrainTrainer(epochs=10) # 10 epochs for initial test
    trainer.train()
