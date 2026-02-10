# =============================================================================
# Core Module
# =============================================================================
"""
Core application logic and orchestration.

Exports:
- TrainingManager: Orchestrate training across pairs
- run_training: Convenience function for training
"""

from core.training_manager import TrainingManager, run_training

__all__ = [
    'TrainingManager',
    'run_training',
]
