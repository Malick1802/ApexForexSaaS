# =============================================================================
# Specialist Model Factory (Phase 3)
# =============================================================================
"""
Autonomous training factory for High-Precision 'Specialist' models.

Orchestrates the lifecycle of model creation:
1. Fetches 3 years of data.
2. Applies Dynamic 1:2 Labeling (Phase 2).
3. Trains Deep LSTM (3-layer) models.
4. Enforces '60% Hard Constraint':
   - If validation accuracy < 60%, triggers hyperparameter search.
   - Retries with adjustable learning rates, units, and dropout.
   - Discards models that fail to meet the standard.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU to avoid GPU hangs
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import random
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow import keras

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from models.enhanced_lstm import build_specialist_lstm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecialistFactory:
    """
    Factory for producing Specialist Models.
    """
    
    def __init__(
        self,
        base_dir: str = "models/specialist",
        history_days: int = 3*365,  # 3 Years
        min_win_rate: float = 0.60
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_days = history_days
        self.min_win_rate = min_win_rate
        
        self.engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        
    def _get_hyperparameter_grid(self) -> List[Dict]:
        """Generate a grid of hyperparameters to try."""
        return [
            # 1. Baseline
            {'lr': 0.001, 'units': 128, 'dropout': 0.3},
            # 2. Lower LR
            {'lr': 0.0005, 'units': 128, 'dropout': 0.3},
            # 3. More Regularization
            {'lr': 0.001, 'units': 128, 'dropout': 0.4},
            # 4. Smaller Model
            {'lr': 0.001, 'units': 64, 'dropout': 0.2},
            # 5. Very slow learning
            {'lr': 0.0001, 'units': 128, 'dropout': 0.2},
        ]

    def train_specialist(self, symbol: str, signal_type: str = "BUY") -> bool:
        """
        Train a specialist model for a specific symbol and signal type.
        
        Returns:
            True if a model meeting the 60% criteria was saved.
        """
        symbol = symbol.upper()
        signal_type = signal_type.upper()
        
        # Check if already trained successfully
        save_dir = self.base_dir / symbol / signal_type
        metrics_path = save_dir / "metrics.json"
        
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    if data.get('accuracy', 0) >= self.min_win_rate:
                        logger.info(f"✅ Skipping {symbol} {signal_type} (Already Certified: {data['accuracy']:.2%})")
                        return True
            except Exception:
                pass # Corrupt file, re-train

        logger.info(f"Starting Specialist Factory for {symbol} ({signal_type})...")
        
        # 1. Fetch Data
        try:
            # We fetch labeled data directly. 
            # Note: fetch_labeled uses the NEW dynamic labeling logic in data_pipeline/labeling.py
            # provided we didn't break the interface.
            df = self.engine.fetch_labeled(symbol, interval="1h", days=self.history_days)
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return False
            
        if len(df) < 500:
            logger.warning(f"Not enough data for {symbol} ({len(df)} rows)")
            return False

        # 2. Prepare Features & Labels
        features = self.feature_engineer.extract_features(df)
        
        # Add Correlated Assets
        correlated = self.engine.get_correlated_assets(symbol)
        if correlated:
            try:
                corr_symbol = correlated[0]['symbol']
                corr_df = self.engine.fetch(corr_symbol, interval="1h", days=self.history_days)
                features = self.feature_engineer.add_correlated_asset(features, corr_df)
            except Exception as e:
                logger.warning(f"Correlation fetch failed: {e}")

        # Label processing for Binary Classification
        # 1=Buy, 2=Sell. 
        # If signal_type="BUY", target is 1. Else 2.
        target_label = 1 if signal_type == "BUY" else 2
        y_all = df['label'].values
        y_binary = (y_all == target_label).astype(int)
        
        # Check imbalance
        pos_rate = y_binary.mean()
        logger.info(f"Positive Rate: {pos_rate:.2%}")
        if pos_rate < 0.05: # Too few signals
            logger.warning("Positive rate too low, skipping.")
            return False

        # Debug: Check for NaNs
        nan_counts = features.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning("Found NaNs in features:")
            logger.warning(nan_counts[nan_counts > 0])
            
        # Create Sequences
        # Pass numpy array to avoid index mismatch with features DataFrame
        X, y = self.feature_engineer.create_sequences(features, y_binary, sequence_length=60)
        
        # Split (Train/Val) - Last 20% for validation (Simulate TimeSeriesSplit final fold)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[2])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_flat = X_val.reshape(-1, X_val.shape[2])
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # 3. Hyperparameter Search Loop
        grid = self._get_hyperparameter_grid()
        
        best_acc = 0.0
        saved = False
        
        for i, params in enumerate(grid):
            logger.info(f"Attempt {i+1}/{len(grid)} with params: {params}")
            
            model = build_specialist_lstm(
                input_shape=(60, X.shape[2]),
                num_classes=1, # Binary
                lstm_units=params['units'],
                dropout_rate=params['dropout'],
                learning_rate=params['lr']
            )
            
            # Class weights
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = dict(zip(classes, weights))
            
            # Train
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=50, # Fixed 50 epochs
                batch_size=32,
                class_weight=class_weight,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
                ],
                verbose=0
            )
            
            # Evaluate
            # Use a threshold of 0.5 for standard accuracy, 
            # OR check if we can get >60% precision with a higher threshold?
            # Prompt says "Validation accuracy ... below 60%".
            # Let's check standard accuracy first.
            loss, acc = model.evaluate(X_val_scaled, y_val, verbose=0)
            
            logger.info(f"Result: Val Accuracy = {acc:.2%}")
            
            if acc >= self.min_win_rate:
                logger.info(f"SUCCESS! Accuracy {acc:.2%} >= 60%. Saving model.")
                self._save_model(model, scaler, symbol, signal_type, acc, params)
                best_acc = acc
                saved = True
                break # Found a winner
            
            if acc > best_acc:
                best_acc = acc
        
        if not saved:
            logger.warning(f"FAILED to meet criteria for {symbol}. Best Accuracy: {best_acc:.2%}")
            return False
            
        return True

    def _save_model(self, model, scaler, symbol, signal_type, accuracy, params):
        """Save the successful specialist model."""
        save_dir = self.base_dir / symbol / signal_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(save_dir / "model.keras")
        joblib.dump(scaler, save_dir / "scaler.joblib")
        
        with open(save_dir / "metrics.json", "w") as f:
            json.dump({
                "symbol": symbol,
                "signal_type": signal_type,
                "accuracy": float(accuracy),
                "params": params,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Specialist model saved to {save_dir}")

    def train_fleet(self):
        """Train both BUY and SELL models for all configured pairs."""
        try:
            with open('config.yaml', 'r') as f:
                import yaml
                config = yaml.safe_load(f)
            
            pairs = []
            for category in ['majors', 'minors', 'crosses']:
                # Config structure is currency_pairs -> majors -> list of dicts
                pair_list = config.get('currency_pairs', {}).get(category, [])
                # Extract symbol string from each dict
                pairs.extend([p['symbol'] for p in pair_list])
            
            # Remove duplicates and sort
            pairs = sorted(list(set(pairs)))
            
            logger.info(f"🚀 Starting Fleet Training for {len(pairs)} pairs...")
            
            results = []
            for symbol in pairs:
                # Train BUY
                buy_success = self.train_specialist(symbol, "BUY")
                results.append({"symbol": symbol, "type": "BUY", "success": buy_success})
                
                # Train SELL 
                sell_success = self.train_specialist(symbol, "SELL")
                results.append({"symbol": symbol, "type": "SELL", "success": sell_success})
                
            logger.info("Fleet Training Complete.")
            df_res = pd.DataFrame(results)
            logger.info(f"Summary:\n{df_res}")
            
        except Exception as e:
            logger.error(f"Fleet training failed: {e}")

if __name__ == "__main__":
    factory = SpecialistFactory()
    # If arguments provided (e.g. specific pair), use them
    if len(sys.argv) > 1:
        pair = sys.argv[1]
        action = sys.argv[2] if len(sys.argv) > 2 else "BUY"
        factory.train_specialist(pair, action)
    else:
        # Default: Run full fleet training as requested
        logger.info("Starting Full Fleet Training sequence...")
        factory.train_fleet()
