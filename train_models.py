#!/usr/bin/env python
# =============================================================================
# LSTM Model Training Script
# =============================================================================
"""
Train LSTM specialist models for all configured currency pairs.

This script orchestrates training across Majors, Minors, and Crosses,
generating a comprehensive report of backtest win rates.

Usage:
    python train_models.py                    # Train all pairs
    python train_models.py --category majors  # Train only majors
    python train_models.py --symbols EURUSD GBPUSD  # Train specific pairs
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.training_manager import TrainingManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Forex Models')
    parser.add_argument(
        '--category',
        choices=['majors', 'minors', 'crosses'],
        help='Train only a specific category'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Train specific symbols (e.g., EURUSD GBPUSD)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs (default: 50)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=50,
        help='LSTM sequence length (default: 50)'
    )
    parser.add_argument(
        '--history-days',
        type=int,
        default=365,
        help='Days of historical data (default: 365)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training with reduced settings for testing'
    )
    
    args = parser.parse_args()
    
    # Training configuration
    if args.quick:
        config = {
            'sequence_length': 30,
            'n_splits': 3,
            'epochs': 10,
            'batch_size': 64,
            'history_days': 180
        }
        logger.info("Using QUICK training mode")
    else:
        config = {
            'sequence_length': args.sequence_length,
            'n_splits': 5,
            'epochs': args.epochs,
            'batch_size': 32,
            'history_days': args.history_days
        }
    
    logger.info(f"Training configuration: {config}")
    
    # Create manager
    manager = TrainingManager(
        base_model_dir='models/trained',
        training_config=config
    )
    
    start_time = datetime.now()
    logger.info(f"Training started at {start_time}")
    
    try:
        if args.symbols:
            # Train specific symbols
            logger.info(f"Training specific symbols: {args.symbols}")
            results = {'custom': manager.train_subset(args.symbols)}
        elif args.category:
            # Train single category
            logger.info(f"Training category: {args.category}")
            results = {args.category: manager.train_category(args.category)}
        else:
            # Train all categories
            logger.info("Training all currency pairs")
            results = manager.train_all()
        
        # Generate and save reports
        report = manager.generate_report(results)
        print("\n" + "="*80)
        print(report)
        print("="*80 + "\n")
        
        manager.save_report()
        manager.save_results_json()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Training completed in {duration:.1f} seconds")
        logger.info(f"Report saved to models/trained/training_report.md")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
