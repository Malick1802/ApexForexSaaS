"""
Enhanced Model Retraining Script
================================
Retrains all currency pair models with improved parameters:
- 729 days of 1h data (yfinance maximum)
- 100 epochs (up from 10)
- 5-fold TimeSeriesSplit CV
- Class weight balancing
- Patience=20 EarlyStopping
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.training_manager import TrainingManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RETRAIN - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "retraining_enhanced.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Retrain")


def main():
    logger.info("=" * 60)
    logger.info("ENHANCED MODEL RETRAINING")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Enhanced training config
    training_config = {
        'epochs': 100,          # Up from 10
        'history_days': 729,    # Max for yfinance 1h data
        'sequence_length': 50,  # Longer context window
        'n_splits': 5,          # More CV folds
        'batch_size': 32,
    }
    
    logger.info(f"Config: {json.dumps(training_config, indent=2)}")
    
    # Initialize manager with enhanced config
    manager = TrainingManager(
        base_model_dir=str(PROJECT_ROOT / "models" / "trained"),
        training_config=training_config
    )
    
    # Train ALL categories
    logger.info("Training all pairs across majors, minors, crosses...")
    results = manager.train_all(categories=['majors', 'minors', 'crosses'])
    
    # Generate and save report
    report = manager.generate_report(results)
    report_path = manager.save_report(
        filepath=str(PROJECT_ROOT / "models" / "trained" / "training_report.md"),
        results=results
    )
    
    # Save JSON results
    json_path = manager.save_results_json(
        filepath=str(PROJECT_ROOT / "models" / "trained" / "training_results.json"),
        results=results
    )
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info(f"RETRAINING COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"Report: {report_path}")
    logger.info(f"Results: {json_path}")
    logger.info("=" * 60)
    
    # Print summary
    print("\n" + report)


if __name__ == "__main__":
    main()
