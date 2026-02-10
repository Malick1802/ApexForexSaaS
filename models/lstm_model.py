# =============================================================================
# LSTM Model Architecture
# =============================================================================
"""
LSTM specialist model for forex signal prediction.

Architecture:
- 2 LSTM layers with dropout
- Dense output with softmax for 3-class classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def build_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    lstm_units: Tuple[int, int] = (64, 32),
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build LSTM model for forex signal classification.
    
    Architecture:
        Input -> LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Output(3)
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of output classes (default: 3 for BUY/SELL/WAIT)
        lstm_units: Units for each LSTM layer
        dropout_rate: Dropout rate after each LSTM layer
        learning_rate: Adam optimizer learning rate
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # First LSTM layer - returns sequences for stacking
    x = layers.LSTM(
        units=lstm_units[0],
        return_sequences=True,
        name='lstm_1'
    )(inputs)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Second LSTM layer - returns final state only
    x = layers.LSTM(
        units=lstm_units[1],
        return_sequences=False,
        name='lstm_2'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Dense layer for feature extraction
    x = layers.Dense(16, activation='relu', name='dense_1')(x)
    
    # Output layer with softmax
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='forex_lstm')
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Built LSTM model: input={input_shape}, classes={num_classes}")
    return model


def build_lstm_with_attention(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    LSTM with self-attention mechanism for enhanced performance.
    
    This is an advanced architecture option for better long-range
    dependency modeling.
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of output classes
        lstm_units: Units for LSTM layer
        dropout_rate: Dropout rate
        learning_rate: Adam learning rate
        
    Returns:
        Compiled Keras model with attention
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # LSTM layer returning full sequence
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=True,
        name='lstm'
    )(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    # Self-attention
    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=16,
        name='attention'
    )(x, x)
    x = layers.Add()([x, attention])  # Residual connection
    x = layers.LayerNormalization()(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='forex_lstm_attention')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(
    model_path: str,
    patience: int = 10,
    min_delta: float = 0.001
) -> list:
    """
    Get training callbacks for model.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    return callbacks


class ConfusionMatrixCallback(keras.callbacks.Callback):
    """
    Callback to log confusion matrix at end of training.
    """
    
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        
    def on_train_end(self, logs=None):
        from sklearn.metrics import confusion_matrix, classification_report
        
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = y_pred.argmax(axis=1)
        
        cm = confusion_matrix(self.y_val, y_pred_classes)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        report = classification_report(
            self.y_val, y_pred_classes,
            target_names=['WAIT', 'BUY', 'SELL'],
            zero_division=0
        )
        logger.info(f"Classification Report:\n{report}")
