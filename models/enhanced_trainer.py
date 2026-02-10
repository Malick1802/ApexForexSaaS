# =============================================================================
# Enhanced Pair Trainer with Win Rate Optimizations
# =============================================================================
"""
Improved training pipeline targeting 60%+ win rate.

Improvements:
- Confidence-based filtering
- Class weighting for imbalanced labels
- Enhanced LSTM architecture
- Binary classification option
- Better metrics tracking
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
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import tensorflow as tf
from tensorflow import keras

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from models.enhanced_lstm import (
    build_enhanced_lstm,
    build_binary_classifier,
    ConfidenceFilter,
    compute_class_weights,
    get_enhanced_callbacks
)


logger = logging.getLogger(__name__)


class EnhancedPairTrainer:
    """
    Enhanced trainer for improved win rates.
    
    Key improvements:
    - Uses enhanced LSTM architecture
    - Applies class weighting
    - Tracks confidence-filtered metrics
    - Supports binary classification mode
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[dict] = None,
        base_model_dir: str = "models/enhanced",
        sequence_length: int = 60,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        history_days: int = 365,
        confidence_threshold: float = 0.7,
        use_class_weights: bool = True,
        binary_mode: bool = False
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            symbol: Currency pair symbol
            config: Optional configuration dict
            base_model_dir: Base directory for saving models
            sequence_length: LSTM sequence length (longer for better patterns)
            n_splits: Number of TimeSeriesSplit folds
            epochs: Training epochs (more for convergence)
            batch_size: Training batch size
            history_days: Days of historical data
            confidence_threshold: Minimum confidence for signals
            use_class_weights: Whether to apply class weighting
            binary_mode: If True, train binary classifiers
        """
        self.symbol = symbol.upper()
        self.config = config or {}
        self.base_model_dir = base_model_dir
        self.sequence_length = sequence_length
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.history_days = history_days
        self.confidence_threshold = confidence_threshold
        self.use_class_weights = use_class_weights
        self.binary_mode = binary_mode
        
        # Model output directory
        self.model_dir = Path(base_model_dir) / self.symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.confidence_filter = ConfidenceFilter(threshold=confidence_threshold)
        
        # Results storage
        self.fold_results: List[dict] = []
        self.final_metrics: dict = {}
        
    def prepare_data(
        self,
        interval: str = "1h"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data with features and labels.
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
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
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
        Train model on a single fold with enhancements.
        """
        logger.info(f"Training fold {fold + 1}/{self.n_splits}...")
        
        # Fit scaler on training data ONLY
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)
        
        # Apply scaler to validation data
        X_val_flat = X_val.reshape(-1, n_features)
        X_val_scaled = fold_scaler.transform(X_val_flat)
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], seq_len, n_features)
        
        # Compute class weights
        class_weight = None
        if self.use_class_weights:
            class_weight = compute_class_weights(y_train)
        
        # Build enhanced model
        model = build_enhanced_lstm(
            input_shape=(seq_len, n_features),
            num_classes=3,
            lstm_units=64,
            attention_heads=4,
            dropout_rate=0.3
        )
        
        # Train with early stopping
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
        
        # Get predictions with probabilities
        y_proba = model.predict(X_val_scaled, verbose=0)
        y_pred_classes = y_proba.argmax(axis=1)
        
        # Calculate standard win rate
        trade_mask = y_pred_classes > 0
        if trade_mask.sum() > 0:
            correct_trades = (y_pred_classes[trade_mask] == y_val[trade_mask]).sum()
            win_rate = correct_trades / trade_mask.sum()
        else:
            win_rate = 0.0
        
        # Calculate FILTERED win rate (high confidence only)
        filter_result = self.confidence_filter.filter_predictions(y_proba, y_val)
        filtered_win_rate = filter_result.get('filtered_accuracy', 0.0)
        n_high_conf = filter_result['n_signals']
        
        metrics = {
            'fold': fold + 1,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'win_rate': float(win_rate),
            'filtered_win_rate': float(filtered_win_rate),
            'n_trades': int(trade_mask.sum()),
            'n_high_confidence': int(n_high_conf),
            'signal_rate': float(filter_result['signal_rate']),
            'epochs_trained': len(history.history['loss'])
        }
        
        logger.info(
            f"Fold {fold + 1}: loss={val_loss:.4f}, "
            f"acc={val_acc:.4f}, win_rate={win_rate:.2%}, "
            f"FILTERED={filtered_win_rate:.2%} ({n_high_conf} signals)"
        )
        
        return model, metrics
    
    def train(self, interval: str = "1h") -> dict:
        """
        Train model with enhanced settings.
        """
        logger.info(f"Starting ENHANCED training for {self.symbol}")
        start_time = datetime.now()
        
        # Prepare data
        X, y, feature_names = self.prepare_data(interval)
        
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        self.fold_results = []
        best_model = None
        best_filtered_win_rate = 0.0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model, metrics = self.train_fold(X_train, y_train, X_val, y_val, fold)
            self.fold_results.append(metrics)
            
            if metrics['filtered_win_rate'] > best_filtered_win_rate:
                best_filtered_win_rate = metrics['filtered_win_rate']
                best_model = model
        
        # Final training on all data
        logger.info("Final training on full dataset...")
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Class weights for final model
        class_weight = None
        if self.use_class_weights:
            class_weight = compute_class_weights(y)
        
        final_model = build_enhanced_lstm(
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
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        # Final evaluation with confidence filtering
        final_proba = final_model.predict(X_val_final, verbose=0)
        final_filter = self.confidence_filter.filter_predictions(final_proba, y_val_final)
        
        # Save model and scaler
        self._save_model(final_model, feature_names)
        
        # Calculate final metrics
        training_time = (datetime.now() - start_time).total_seconds()
        
        avg_win_rate = np.mean([r['win_rate'] for r in self.fold_results])
        avg_accuracy = np.mean([r['val_accuracy'] for r in self.fold_results])
        avg_filtered_win_rate = np.mean([r['filtered_win_rate'] for r in self.fold_results])
        
        self.final_metrics = {
            'symbol': self.symbol,
            'training_time_seconds': training_time,
            'n_samples': int(len(X)),
            'n_features': int(n_features),
            'sequence_length': self.sequence_length,
            'n_folds': self.n_splits,
            'confidence_threshold': self.confidence_threshold,
            
            # Standard metrics
            'avg_win_rate': float(avg_win_rate),
            'avg_accuracy': float(avg_accuracy),
            
            # FILTERED metrics (high confidence)
            'avg_filtered_win_rate': float(avg_filtered_win_rate),
            'best_filtered_win_rate': float(best_filtered_win_rate),
            
            # Final model metrics
            'final_filtered_win_rate': float(final_filter.get('filtered_accuracy', 0)),
            'final_signal_rate': float(final_filter['signal_rate']),
            
            'fold_results': self.fold_results,
            'trained_at': datetime.now().isoformat()
        }
        
        # Save metrics
        metrics_path = self.model_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.final_metrics, f, indent=2)
        
        logger.info(
            f"Training complete for {self.symbol}: "
            f"avg_win_rate={avg_win_rate:.2%}, "
            f"FILTERED_win_rate={avg_filtered_win_rate:.2%}"
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
        
        # Save config
        config = {
            'symbol': self.symbol,
            'sequence_length': self.sequence_length,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'confidence_threshold': self.confidence_threshold,
            'model_type': 'enhanced_lstm',
            'created_at': datetime.now().isoformat()
        }
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


class BinaryPairTrainer:
    """
    Binary classifier trainer for higher accuracy.
    
    Trains separate BUY and SELL classifiers, which typically
    achieve higher accuracy than 3-class classification.
    """
    
    def __init__(
        self,
        symbol: str,
        signal_type: str = "BUY",  # "BUY" or "SELL"
        base_model_dir: str = "models/binary",
        sequence_length: int = 60,
        epochs: int = 100,
        history_days: int = 365
    ):
        self.symbol = symbol.upper()
        self.signal_type = signal_type.upper()
        self.base_model_dir = base_model_dir
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.history_days = history_days
        
        # Output directory
        self.model_dir = Path(base_model_dir) / self.symbol / self.signal_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
    
    def prepare_binary_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert 3-class labels to binary."""
        target_class = 1 if self.signal_type == "BUY" else 2
        return (y == target_class).astype(int)
    
    def train(self, interval: str = "1h") -> dict:
        """Train binary classifier."""
        logger.info(f"Training {self.signal_type} classifier for {self.symbol}")
        
        # Fetch data
        df = self.engine.fetch_labeled(self.symbol, interval, days=self.history_days)
        features = self.feature_engineer.extract_features(df)
        labels = df['label']
        
        X, y = self.feature_engineer.create_sequences(
            features, labels, self.sequence_length
        )
        
        # Convert to binary
        y_binary = self.prepare_binary_labels(y)
        logger.info(f"Binary label distribution: {dict(zip(*np.unique(y_binary, return_counts=True)))}")
        
        # Scale features
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Train/val split
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_binary[:split_idx], y_binary[split_idx:]
        
        # Class weights for imbalance
        class_weight = compute_class_weights(y_train)
        
        # Build and train
        model = build_binary_classifier(
            input_shape=(seq_len, n_features),
            signal_type=self.signal_type
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=15,
                    restore_best_weights=True,
                    mode='max',
                    verbose=0
                )
            ],
            verbose=0
        )
        
        # Evaluate
        results = model.evaluate(X_val, y_val, verbose=0)
        val_loss, val_acc, val_auc = results
        
        # Predictions
        y_pred_proba = model.predict(X_val, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # High confidence accuracy
        high_conf_mask = (y_pred_proba > 0.7) | (y_pred_proba < 0.3)
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_pred[high_conf_mask] == y_val[high_conf_mask]).mean()
        else:
            high_conf_acc = 0.0
        
        metrics = {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'val_accuracy': float(val_acc),
            'val_auc': float(val_auc),
            'high_confidence_accuracy': float(high_conf_acc),
            'n_samples': int(len(X)),
            'positive_rate': float(y_binary.mean())
        }
        
        # Save
        model.save(self.model_dir / 'model.keras')
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
        
        with open(self.model_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(
            f"{self.signal_type} classifier: acc={val_acc:.2%}, "
            f"AUC={val_auc:.4f}, high_conf_acc={high_conf_acc:.2%}"
        )
        
        return metrics


def train_enhanced(symbol: str, **kwargs) -> dict:
    """Convenience function for enhanced training."""
    trainer = EnhancedPairTrainer(symbol, **kwargs)
    return trainer.train()


def train_binary_pair(symbol: str) -> Dict[str, dict]:
    """Train both BUY and SELL classifiers for a pair."""
    buy_trainer = BinaryPairTrainer(symbol, signal_type="BUY")
    sell_trainer = BinaryPairTrainer(symbol, signal_type="SELL")
    
    return {
        'BUY': buy_trainer.train(),
        'SELL': sell_trainer.train()
    }
