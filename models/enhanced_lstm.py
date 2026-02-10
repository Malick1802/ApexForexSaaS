# =============================================================================
# Enhanced LSTM Model Architecture
# =============================================================================
"""
Advanced LSTM architecture for improved forex signal prediction.

Improvements over base model:
- Bidirectional LSTM for forward/backward context
- Multi-head attention for pattern recognition
- Residual connections for gradient flow
- Batch normalization for training stability
- Configurable confidence thresholds
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict
import numpy as np
import logging


logger = logging.getLogger(__name__)


def build_enhanced_lstm(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    lstm_units: int = 64,
    attention_heads: int = 4,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    use_bidirectional: bool = True,
    use_attention: bool = True
) -> keras.Model:
    """
    Enhanced LSTM with attention and bidirectional processing.
    
    Architecture:
        Input -> BiLSTM -> Attention -> Dense -> Output
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of output classes
        lstm_units: Units for LSTM layer
        attention_heads: Number of attention heads
        dropout_rate: Dropout rate
        learning_rate: Adam learning rate
        use_bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to use attention mechanism
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Batch normalization on input
    x = layers.BatchNormalization(name='input_bn')(inputs)
    
    # First LSTM layer
    lstm_layer = layers.LSTM(
        units=lstm_units,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='lstm_1'
    )
    
    if use_bidirectional:
        x = layers.Bidirectional(lstm_layer, name='bilstm_1')(x)
    else:
        x = lstm_layer(x)
    
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    
    # Second LSTM layer
    lstm_layer_2 = layers.LSTM(
        units=lstm_units // 2,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='lstm_2'
    )
    
    if use_bidirectional:
        x = layers.Bidirectional(lstm_layer_2, name='bilstm_2')(x)
    else:
        x = lstm_layer_2(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    # Multi-head attention
    if use_attention:
        attention_output = layers.MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=lstm_units // attention_heads,
            dropout=dropout_rate,
            name='attention'
        )(x, x)
        
        # Residual connection
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with residual
    dense_1 = layers.Dense(64, activation='relu', name='dense_1')(x)
    dense_1 = layers.Dropout(dropout_rate)(dense_1)
    
    dense_2 = layers.Dense(32, activation='relu', name='dense_2')(dense_1)
    dense_2 = layers.Dropout(dropout_rate / 2)(dense_2)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(dense_2)
    
    model = Model(inputs=inputs, outputs=outputs, name='enhanced_lstm')
    
    # Compile with Adam optimizer and label smoothing
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    logger.info(f"Built enhanced LSTM: input={input_shape}, classes={num_classes}")
    return model


def build_binary_classifier(
    input_shape: Tuple[int, int],
    signal_type: str = "BUY",
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Binary classifier for simpler, higher-accuracy predictions.
    
    Instead of 3-class, predicts: "Is this a good {signal_type} opportunity?"
    
    Args:
        input_shape: (sequence_length, num_features)
        signal_type: "BUY" or "SELL"
        lstm_units: Units for LSTM
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        
    Returns:
        Binary classification model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Batch normalization
    x = layers.BatchNormalization()(inputs)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=False)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Binary output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f'{signal_type.lower()}_classifier')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    logger.info(f"Built {signal_type} binary classifier: input={input_shape}")
    return model


class ConfidenceFilter:
    """
    Filter predictions by confidence threshold.
    
    Only returns signals where model confidence exceeds threshold,
    improving win rate at the cost of fewer signals.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize confidence filter.
        
        Args:
            threshold: Minimum softmax probability for signal
        """
        self.threshold = threshold
        self.stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'filtered_correct': 0,
            'filtered_total': 0
        }
    
    def filter_predictions(
        self,
        probabilities: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Filter predictions by confidence.
        
        Args:
            probabilities: Softmax probabilities (N, num_classes)
            true_labels: Optional true labels for accuracy calc
            
        Returns:
            Dict with filtered predictions and stats
        """
        # Get predicted classes and confidences
        predicted_classes = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        
        # Apply threshold
        high_conf_mask = confidences >= self.threshold
        
        # Only consider trading signals (not WAIT=0)
        trade_mask = predicted_classes > 0
        
        # Combined mask: high confidence AND trading signal
        filtered_mask = high_conf_mask & trade_mask
        
        self.stats['total_predictions'] += len(probabilities)
        self.stats['high_confidence'] += filtered_mask.sum()
        
        result = {
            'predictions': predicted_classes,
            'confidences': confidences,
            'filtered_mask': filtered_mask,
            'filtered_predictions': predicted_classes[filtered_mask],
            'filtered_confidences': confidences[filtered_mask],
            'n_signals': filtered_mask.sum(),
            'signal_rate': filtered_mask.mean()
        }
        
        # Calculate filtered accuracy if labels provided
        if true_labels is not None:
            if filtered_mask.sum() > 0:
                filtered_correct = (
                    predicted_classes[filtered_mask] == true_labels[filtered_mask]
                ).sum()
                filtered_accuracy = filtered_correct / filtered_mask.sum()
                
                self.stats['filtered_correct'] += filtered_correct
                self.stats['filtered_total'] += filtered_mask.sum()
                
                result['filtered_accuracy'] = filtered_accuracy
                result['filtered_correct'] = filtered_correct
            else:
                result['filtered_accuracy'] = 0.0
                result['filtered_correct'] = 0
        
        return result
    
    def get_overall_stats(self) -> Dict:
        """Get cumulative statistics."""
        if self.stats['filtered_total'] > 0:
            overall_accuracy = (
                self.stats['filtered_correct'] / self.stats['filtered_total']
            )
        else:
            overall_accuracy = 0.0
        
        return {
            'total_predictions': self.stats['total_predictions'],
            'high_confidence_signals': self.stats['high_confidence'],
            'signal_rate': (
                self.stats['high_confidence'] / max(1, self.stats['total_predictions'])
            ),
            'filtered_win_rate': overall_accuracy
        }


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced data.
    
    Args:
        labels: Array of integer labels
        
    Returns:
        Dict mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Computed class weights: {weight_dict}")
    
    return weight_dict


def get_enhanced_callbacks(
    model_path: str,
    patience: int = 15,
    reduce_lr_patience: int = 7
) -> list:
    """
    Get enhanced training callbacks.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        reduce_lr_patience: LR reduction patience
        
    Returns:
        List of Keras callbacks
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TerminateOnNaN()
    ]


def build_specialist_lstm(
    input_shape: Tuple[int, int],
    num_classes: int = 1, # Default to binary (sigmoid) or 3 for multiclass
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Deep LSTM Model (3 layers) for Specialist Factory (Phase 3).
    
    Structure:
    - Input -> BN
    - LSTM (units) + BN + Dropout
    - LSTM (units) + BN + Dropout
    - LSTM (units/2) + BN + Dropout
    - Dense Output
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    
    # Layer 1
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Layer 2
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Layer 3
    x = layers.LSTM(lstm_units // 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'sparse_categorical_crossentropy'
        
    outputs = layers.Dense(num_classes, activation=activation)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='specialist_lstm')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model
