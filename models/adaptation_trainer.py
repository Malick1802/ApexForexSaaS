import os
import gc
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import yaml
import joblib

logger = logging.getLogger(__name__)

# Ensure Eager Execution is enabled for tensor conversion
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

class AdaptationTrainer:
    """
    Phase 3: Expert Adaptation (Transfer Learning)
    Takes the Phase 2 Global Brain and fine-tunes it for individual pairs.
    """
    def __init__(self, config_path: str = "config.yaml"):
        # Setup paths
        self.project_root = Path(os.getcwd())
        self.config_path = self.project_root / config_path
        self.model_dir = self.project_root / "models" / "foundation"
        self.expert_dir = self.project_root / "models" / "expert"
        self.data_dir = Path("/tmp/global_training")
        
        self.expert_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Get all pairs from all categories (Majors, Minors, Crosses, Indices, Commodities)
        self.symbols = []
        if 'currency_pairs' in self.config:
            for cat in self.config['currency_pairs']:
                for item in self.config['currency_pairs'][cat]:
                    self.symbols.append(item['symbol'])
        
        # Remove duplicates
        self.symbols = list(set(self.symbols))
        logger.info(f"Trainer initialized with {len(self.symbols)} total symbols.")
                
    def run_adaptation(self):
        # 1. Load Foundation Model & Scaler
        brain_path = self.model_dir / "foundation_brain.keras"
        scaler_path = self.model_dir / "scaler.joblib"
        
        if not brain_path.exists() or not scaler_path.exists():
            logger.error("Foundation model or scaler not found! Phase 2 must be complete.")
            return

        logger.info(f"Loading Global Foundation Brain from {brain_path}")
        
        from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork
        global_model = keras.models.load_model(
            str(brain_path),
            custom_objects={
                'VariableSelectionNetwork': VariableSelectionNetwork,
                'GatedResidualNetwork': GatedResidualNetwork
            }
        )
        
        scaler = joblib.load(str(scaler_path))
        mean = scaler.mean_.astype(np.float32)
        scale = scaler.scale_.astype(np.float32)

        from data_pipeline.providers.mt5_provider import MT5Provider
        from data_pipeline.engine import DataEngine
        from data_pipeline.features import FeatureEngineer
        from data_pipeline.global_features import GlobalFeatureEngineer
        from data_pipeline.labeling import triple_barrier_label
        import pandas as pd
        
        data_engine = DataEngine()
        feature_engineer = FeatureEngineer()
        global_engineer = GlobalFeatureEngineer()

        # Pre-fetch Global Context for intelligence matrix
        global_data = {}
        for s in ["GOLD", "^TNX"]:
            try:
                df = data_engine.fetch(s, interval="1h", days=1825)
                if df is not None and not df.empty:
                    global_data[s] = df
            except: pass

        # 2. Iterate and Adapt
        for symbol in self.symbols:
            # Check if Expert already exists to avoid redundant training
            expert_model_path = self.expert_dir / symbol / "expert_model.keras"
            if expert_model_path.exists():
                logger.info(f"Skipping {symbol} (Expert already exists)")
                continue

            logger.info(f"========== Adapting Expert for {symbol} ==========")
            
            try:
                # Dynamically load data from local SQLite cache instead of volatile /tmp/ files
                df = data_engine.fetch(symbol, interval="1h", days=1825)
                if df is None or len(df) < 500:
                    logger.warning(f"Not enough historical data cached for {symbol}. Skipping.")
                    continue
                    
                df_labeled = triple_barrier_label(df, stop_loss_pips=25, symbol=symbol)
                base_features = feature_engineer.extract_features(df_labeled)
                features = global_engineer.add_global_features(symbol, base_features, global_data)
                
                y = df_labeled['label'].astype(int).values
                X, y_seq = feature_engineer.create_sequences(features, y, sequence_length=60)
                
                if len(X) == 0:
                    logger.warning(f"Feature sequence generation failed for {symbol}.")
                    continue
                    
                split = int(len(X) * 0.90)
                X_train, y_train = X[:split], y_seq[:split]
                X_val, y_val = X[split:], y_seq[split:]
                
                # Apply scaler
                X_train_flat = X_train.reshape(-1, X_train.shape[2])
                X_train_flat = (X_train_flat - mean) / scale
                X_train = X_train_flat.reshape(X_train.shape)
                
                X_val_flat = X_val.reshape(-1, X_val.shape[2])
                X_val_flat = (X_val_flat - mean) / scale
                X_val = X_val_flat.reshape(X_val.shape)
                
                # Create a fresh copy of the model for this pair
                # Using clone_model ensures we don't bleed weights across loops
                pair_model = keras.models.clone_model(global_model)
                pair_model.set_weights(global_model.get_weights())
                
                # Freeze Core Layers (Keep TFT/LSTM locked)
                # We only unfreeze the final Dense output and maybe the GRN right before it
                for layer in pair_model.layers:
                    if layer.name == 'output' or 'gated_residual_network' in layer.name:
                        layer.trainable = True
                    else:
                        layer.trainable = False
                        
                # Recompile with very low learning rate
                pair_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                logger.info(f"Training {symbol} Expert (Samples: {len(X_train)})...")
                class_weight = {0: 0.5, 1: 2.0, 2: 2.0}
                
                pair_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10, # Fast fine-tuning
                    batch_size=128,
                    class_weight=class_weight,
                    callbacks=[
                        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
                    ],
                    verbose=1
                )
                
                # Save Expert Model
                expert_pair_dir = self.expert_dir / symbol
                expert_pair_dir.mkdir(exist_ok=True)
                
                pair_model.save(str(expert_pair_dir / "expert_model.keras"))
                
                # Save config with volume info
                with open(expert_pair_dir / "config.json", 'w') as f:
                    json.dump({
                        'trades': int(len(X_train)),
                        'base_accuracy': float(pair_model.evaluate(X_val, y_val, verbose=0)[1])
                    }, f)
                
                logger.info(f"✅ {symbol} Expert Adaptation Complete!")
                
                # Memory Safeguard
                del pair_model, X_train, y_train, X_val, y_val, X, y_seq
                keras.backend.clear_session()
                gc.collect()
                
                # Free memory
                del X_train, y_train, X_val, y_val, pair_model
                keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to adapt {symbol}: {e}")
                
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        trainer = AdaptationTrainer()
        trainer.run_adaptation()
    except Exception as e:
        logger.error(f"Adaptation failed: {e}")
