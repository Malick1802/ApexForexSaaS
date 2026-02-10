# =============================================================================
# Specialist Pair Trainer
# =============================================================================
"""
Trainer for "Specialist" models using a specific LSTM architecture.

Architecture:
- 2 Hidden Layers
- Dropout (0.2)
- Features: OHLCV + RSI + ATR + Correlated Asset
"""

import logging
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from models.enhanced_trainer import EnhancedPairTrainer
from models.lstm_model import build_lstm_model

logger = logging.getLogger(__name__)


def compute_class_weights(labels: np.ndarray) -> dict:
    """Helper to compute class weights."""
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


class SpecialistPairTrainer(EnhancedPairTrainer):
    """
    Specialist trainer extending the enhanced pipeline but with
    specific architecture requirements.
    """
    
    def __init__(self, symbol: str, **kwargs):
        # Enforce specific defaults for Specialist models if not provided
        kwargs.setdefault('base_model_dir', 'models/specialist')
        super().__init__(symbol, **kwargs)
        
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold: int
    ) -> tuple:
        """
        Override train_fold to use build_lstm_model instead of enhanced_lstm.
        """
        logger.info(f"Training Specialist fold {fold + 1}/{self.n_splits}...")
        
        # Fit scaler on training data ONLY (Strict requirement)
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)
        
        # Transform validation
        X_val_flat = X_val.reshape(-1, n_features)
        X_val_scaled = fold_scaler.transform(X_val_flat)
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], seq_len, n_features)
        
        # Class weights
        class_weight = None
        if self.use_class_weights:
            class_weight = compute_class_weights(y_train)
            
        # Build Specialist Model (2 layers, 0.2 dropout)
        model = build_lstm_model(
            input_shape=(seq_len, n_features),
            num_classes=3,
            lstm_units=(64, 32), # 2 Hidden layers
            dropout_rate=0.2     # Specified dropout
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20, # Increased for 5-year data
                    restore_best_weights=True,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
        
        # Metrics logic reused from parent, but repeated here for clarity/access
        y_proba = model.predict(X_val_scaled, verbose=0)
        y_pred_classes = y_proba.argmax(axis=1)
        
        trade_mask = y_pred_classes > 0
        if trade_mask.sum() > 0:
            correct_trades = (y_pred_classes[trade_mask] == y_val[trade_mask]).sum()
            win_rate = correct_trades / trade_mask.sum()
        else:
            win_rate = 0.0
            
        # Filtered metrics
        filter_result = self.confidence_filter.filter_predictions(y_proba, y_val)
        filtered_win_rate = filter_result.get('filtered_accuracy', 0.0)
        
        metrics = {
            'fold': fold + 1,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'win_rate': float(win_rate),
            'filtered_win_rate': float(filtered_win_rate),
            'n_trades': int(trade_mask.sum()),
            'epochs_trained': len(history.history['loss'])
        }
        
        logger.info(
            f"Fold {fold + 1}: acc={val_acc:.4f}, win_rate={win_rate:.2%}, "
            f"FILTERED={filtered_win_rate:.2%}"
        )
        
        # Return model and metrics
        return model, metrics
