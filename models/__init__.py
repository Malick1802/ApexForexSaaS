# =============================================================================
# Models Module
# =============================================================================
"""
LSTM models for forex signal prediction.

Exports:
- build_lstm_model: Create LSTM architecture
- build_enhanced_lstm: Enhanced LSTM with attention
- PairTrainer: Train model for single pair
- EnhancedPairTrainer: Train with confidence filtering
- BinaryPairTrainer: Binary classification for higher accuracy
- ConfidenceFilter: Filter predictions by confidence
"""

from models.lstm_model import (
    build_lstm_model,
    build_lstm_with_attention,
    get_callbacks,
)
from models.trainer import PairTrainer, train_single_pair
from models.enhanced_lstm import (
    build_enhanced_lstm,
    build_binary_classifier,
    ConfidenceFilter,
    compute_class_weights,
)
from models.enhanced_trainer import (
    EnhancedPairTrainer,
    BinaryPairTrainer,
    train_enhanced,
    train_binary_pair,
)

__all__ = [
    # Base models
    'build_lstm_model',
    'build_lstm_with_attention',
    'get_callbacks',
    'PairTrainer',
    'train_single_pair',
    
    # Enhanced models
    'build_enhanced_lstm',
    'build_binary_classifier',
    'ConfidenceFilter',
    'compute_class_weights',
    'EnhancedPairTrainer',
    'BinaryPairTrainer',
    'train_enhanced',
    'train_binary_pair',
]
