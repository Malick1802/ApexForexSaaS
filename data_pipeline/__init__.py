# =============================================================================
# Data Pipeline Module
# =============================================================================
"""
Forex data fetching, processing, and labeling pipeline.

Main exports:
- DataEngine: Main data orchestration class
- triple_barrier_label: Labeling function for ML training
- DataProviderBase: Abstract base for custom providers
"""

from data_pipeline.engine import DataEngine, create_engine
from data_pipeline.labeling import (
    triple_barrier_label,
    triple_barrier_label_vectorized,
    get_pip_value,
    validate_no_lookahead,
)
from data_pipeline.base import DataProviderBase

__all__ = [
    "DataEngine",
    "create_engine",
    "triple_barrier_label",
    "triple_barrier_label_vectorized",
    "get_pip_value",
    "validate_no_lookahead",
    "DataProviderBase",
]
