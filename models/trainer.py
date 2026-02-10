# =============================================================================
# Single Pair Trainer
# =============================================================================
"""
Training pipeline for individual currency pair LSTM models.

Handles:
- Data fetching and feature extraction
- TimeSeriesSplit cross-validation
- Scaler fitting (training data only)
- Model training and evaluation
- Model and scaler persistence
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras

# Local imports
from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from models.lstm_model import build_lstm_model, get_callbacks


logger = logging.getLogger(__name__)


class PairTrainer:
    """
    Trainer for a single currency pair LSTM model.
    
    Implements proper time-series cross-validation with
    scaler fitting only on training data to prevent look-ahead bias.
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[dict] = None,
        base_model_dir: str = "models",
        sequence_length: int = 50,
        n_splits: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        history_days: int = 365
    ):
        """
        Initialize pair trainer.
        
        Args:
            symbol: Currency pair symbol (e.g., "EURUSD")
            config: Optional configuration dict
            base_model_dir: Base directory for saving models
            sequence_length: LSTM sequence length
            n_splits: Number of TimeSeriesSplit folds
            epochs: Training epochs per fold
            batch_size: Training batch size
            history_days: Days of historical data to use
        """
        self.symbol = symbol.upper()
        self.config = config or {}
        self.base_model_dir = base_model_dir
        self.sequence_length = sequence_length
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.history_days = history_days
        
        # Model output directory
        self.model_dir = Path(base_model_dir) / self.symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        
        # Results storage
        self.fold_results: List[dict] = []
        self.final_metrics: dict = {}
        
    def prepare_data(
        self,
        interval: str = "1h"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data with features and labels.
        
        Args:
            interval: Data interval
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info(f"Preparing data for {self.symbol}...")
        
        # Fetch labeled data
        df = self.engine.fetch_labeled(
            self.symbol,
            interval=interval,
            days=self.history_days
        )
        
        # Extract features
        features = self.feature_engineer.extract_features(df)
        
        # Try to add correlated asset
        correlated = self.engine.get_correlated_assets(self.symbol)
        if correlated:
            try:
                # Get the first correlated asset
                corr_symbol = correlated[0]['symbol']
                corr_df = self.engine.fetch(corr_symbol, interval, days=self.history_days)
                features = self.feature_engineer.add_correlated_asset(
                    features, corr_df, 'corr'
                )
            except Exception as e:
                logger.warning(f"Could not fetch correlated asset: {e}")
        
        # Get labels
        labels = df['label']
        
        # Create sequences
        X, y = self.feature_engineer.create_sequences(
            features, labels, self.sequence_length
        )
        
        feature_names = list(features.columns)
        
        logger.info(f"Data prepared: X={X.shape}, y={y.shape}")
        return X, y, feature_names
    
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold: int
    ) -> Tuple[keras.Model, dict]:
        """
        Train model on a single fold.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            fold: Fold number
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training fold {fold + 1}/{self.n_splits}...")
        
        # Fit scaler on training data ONLY
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)
        
        # Apply scaler to validation data (transform only, no fit!)
        X_val_flat = X_val.reshape(-1, n_features)
        X_val_scaled = fold_scaler.transform(X_val_flat)
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], seq_len, n_features)
        
        # Build model
        model = build_lstm_model(
            input_shape=(seq_len, n_features),
            num_classes=3
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
        
        # Calculate win rate (trades that hit TP)
        y_pred = model.predict(X_val_scaled, verbose=0)
        y_pred_classes = y_pred.argmax(axis=1)
        
        # Win rate: when model predicts BUY/SELL, how often is it correct?
        trade_mask = y_pred_classes > 0  # BUY or SELL predictions
        if trade_mask.sum() > 0:
            correct_trades = (y_pred_classes[trade_mask] == y_val[trade_mask]).sum()
            win_rate = correct_trades / trade_mask.sum()
        else:
            win_rate = 0.0
        
        metrics = {
            'fold': fold + 1,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'win_rate': float(win_rate),
            'n_trades': int(trade_mask.sum()),
            'epochs_trained': len(history.history['loss'])
        }
        
        logger.info(
            f"Fold {fold + 1}: loss={val_loss:.4f}, "
            f"acc={val_acc:.4f}, win_rate={win_rate:.2%}"
        )
        
        return model, metrics
    
    def train(self, interval: str = "1h") -> dict:
        """
        Train model with TimeSeriesSplit cross-validation.
        
        Args:
            interval: Data interval
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {self.symbol}")
        start_time = datetime.now()
        
        # Prepare data
        X, y, feature_names = self.prepare_data(interval)
        
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        self.fold_results = []
        best_model = None
        best_win_rate = 0.0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model, metrics = self.train_fold(X_train, y_train, X_val, y_val, fold)
            self.fold_results.append(metrics)
            
            if metrics['win_rate'] > best_win_rate:
                best_win_rate = metrics['win_rate']
                best_model = model
        
        # Final training on all data
        logger.info("Final training on full dataset...")
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        final_model = build_lstm_model(
            input_shape=(seq_len, n_features),
            num_classes=3
        )
        
        # Split last 10% for validation
        split_idx = int(len(X_scaled) * 0.9)
        X_train_final = X_scaled[:split_idx]
        y_train_final = y[:split_idx]
        X_val_final = X_scaled[split_idx:]
        y_val_final = y[split_idx:]
        
        final_model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        # Save model and scaler
        self._save_model(final_model, feature_names)
        
        # Calculate final metrics
        training_time = (datetime.now() - start_time).total_seconds()
        
        avg_win_rate = np.mean([r['win_rate'] for r in self.fold_results])
        avg_accuracy = np.mean([r['val_accuracy'] for r in self.fold_results])
        
        self.final_metrics = {
            'symbol': self.symbol,
            'training_time_seconds': training_time,
            'n_samples': int(len(X)),
            'n_features': int(n_features),
            'sequence_length': self.sequence_length,
            'n_folds': self.n_splits,
            'avg_win_rate': float(avg_win_rate),
            'avg_accuracy': float(avg_accuracy),
            'best_fold_win_rate': float(best_win_rate),
            'fold_results': self.fold_results,
            'trained_at': datetime.now().isoformat()
        }
        
        # Save metrics
        metrics_path = self.model_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.final_metrics, f, indent=2)
        
        logger.info(
            f"Training complete for {self.symbol}: "
            f"avg_win_rate={avg_win_rate:.2%}, avg_acc={avg_accuracy:.4f}"
        )
        
        return self.final_metrics
    
    def _save_model(self, model: keras.Model, feature_names: List[str]) -> None:
        """Save model, scaler, and config."""
        # Save model
        model_path = self.model_dir / 'model.keras'
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save config
        config = {
            'symbol': self.symbol,
            'sequence_length': self.sequence_length,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'created_at': datetime.now().isoformat()
        }
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model(self) -> Tuple[keras.Model, StandardScaler, dict]:
        """
        Load saved model, scaler, and config.
        
        Returns:
            Tuple of (model, scaler, config)
        """
        model_path = self.model_dir / 'model.keras'
        scaler_path = self.model_dir / 'scaler.joblib'
        config_path = self.model_dir / 'config.json'
        
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return model, scaler, config


def train_single_pair(symbol: str, **kwargs) -> dict:
    """
    Convenience function to train a single pair.
    
    Args:
        symbol: Currency pair symbol
        **kwargs: Additional arguments for PairTrainer
        
    Returns:
        Training results
    """
    trainer = PairTrainer(symbol, **kwargs)
    return trainer.train()
