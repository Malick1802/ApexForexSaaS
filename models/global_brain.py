import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional

class GatedResidualNetwork(layers.Layer):
    """
    GRN from the TFT paper. 
    Controls the flow of information; can act as a simple nonlinear layer or a pass-through.
    """
    def __init__(self, units: int, dropout_rate: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_1 = layers.Dense(units)
        self.dense_2 = layers.Dense(units)
        self.glu = layers.Dense(units, activation='sigmoid') # Gated Linear Unit
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        self.project_input = layers.Dense(units) # For skip connection

    def call(self, x, training=False):
        # Linear layer with ELU activation
        h = layers.Activation('elu')(self.dense_1(x))
        h = self.dense_2(h)
        h = self.dropout(h, training=training)
        
        # Gating
        g = self.glu(h)
        gate_output = x if x.shape[-1] == self.units else self.project_input(x)
        
        # Skip connection + Norm
        return self.layer_norm(gate_output + (g * h))

class VariableSelectionNetwork(layers.Layer):
    """
    VSN from the TFT paper.
    Learns to weigh the importance of each input feature.
    """
    def __init__(self, num_features: int, units: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.grns = [GatedResidualNetwork(units) for _ in range(num_features)]
        self.selector_grn = GatedResidualNetwork(units)
        self.softmax = layers.Dense(num_features, activation='softmax')

    def call(self, x, training=False):
        # x shape: (batch, time, num_features)
        # Create a list of GRN outputs for each feature
        # (This is a simplified version; real VSN often processes features independently)
        feature_embeddings = []
        for i in range(self.num_features):
            # Extract 1 feature dimension and expand to units
            feat = tf.expand_dims(x[..., i], axis=-1)
            feature_embeddings.append(self.grns[i](feat, training=training))
        
        # Stack embeddings: (batch, time, num_features, units)
        stacked = tf.stack(feature_embeddings, axis=-2)
        
        # Compute weights
        # Flatten time and feature dims for the selector grn
        v = self.selector_grn(layers.Flatten()(x) if len(x.shape) == 2 else x, training=training)
        weights = self.softmax(v) # (batch, time, num_features)
        
        # Weighted sum
        weights = tf.expand_dims(weights, axis=-1) # (batch, time, num_features, 1)
        return tf.reduce_sum(weights * stacked, axis=-2)

class TFTModel:
    """
    A simplified Temporal Fusion Transformer for Forex Prediction.
    
    Architecture:
    Input -> VSN -> SeqEncoding (Stacked LSTM/BiLSTM) -> Multi-Head Attention -> GRN -> Dense Output
    """
    def __init__(
        self, 
        input_shape: Tuple[int, int], 
        num_classes: int = 1,
        units: int = 64,
        num_heads: int = 4
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.units = units
        self.num_heads = num_heads

    def build(self) -> Model:
        inputs = layers.Input(shape=self.input_shape, name='market_data')
        
        # 1. Variable Selection (Intelligence layer: decides which indicators matter)
        x = VariableSelectionNetwork(num_features=self.input_shape[1], units=self.units)(inputs)
        
        # 2. Sequential Encoding (Context layer)
        x = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True))(x)
        x = layers.Dropout(0.4)(x)  # Aggressive dropout to close train/val gap
        x = layers.LayerNormalization()(x)
        
        # 3. Temporal Fusion (Multi-Head Attention - Phase 1: Local)
        attention_1 = layers.MultiHeadAttention(
            num_heads=8, # Increased from 4
            key_dim=self.units // 8
        )(x, x)
        x = layers.Add()([x, attention_1])
        x = layers.LayerNormalization()(x)
        
        # 3b. Temporal Fusion (Multi-Head Attention - Phase 2: Global)
        # Deepens the temporal understanding
        attention_2 = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=self.units // 8
        )(x, x)
        x = layers.Add()([x, attention_2])
        x = layers.Dropout(0.3)(x)
        x = layers.LayerNormalization()(x)
        
        # 4. Gated Residual Flow
        x = GatedResidualNetwork(self.units)(x)
        
        # 5. Output Processing
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final "Global Wisdom" Bottleneck (GELU powered)
        x = layers.Dense(self.units * 2, activation='gelu')(x)
        x = layers.Dropout(0.2)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
            
        model = Model(inputs=inputs, outputs=outputs, name='GlobalBrain_TFT_Ultimate')
        
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.0003, weight_decay=0.004),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model

def build_global_brain(input_shape, num_classes=1, units=64):
    """Factory function for the Global Brain."""
    tft = TFTModel(input_shape, num_classes, units)
    return tft.build()
