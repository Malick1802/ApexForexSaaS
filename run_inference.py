#!/usr/bin/env python
"""
Inference Service - Autonomous Signal Generation
==================================================

Runs the inference engine in the background to continuously monitor
currency pairs and generate trading signals automatically.

Usage:
    python run_inference.py --interval 5
    python run_inference.py --symbols EURUSD GBPUSD
    python run_inference.py --confidence 0.70
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.inference import InferenceEngine


def setup_logging(log_file: str = "inference.log"):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run autonomous forex signal generation service"
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Minutes between signal scans (default: 5)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Specific symbols to monitor (default: all configured pairs)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.65,
        help='Minimum confidence threshold for signals (default: 0.65)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/binary',
        help='Directory containing trained models (default: models/binary)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='inference.log',
        help='Log file path (default: inference.log)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("FOREX INFERENCE SERVICE - STARTING")
    logger.info("=" * 70)
    logger.info(f"Model Directory: {args.model_dir}")
    logger.info(f"Scan Interval: {args.interval} minutes")
    logger.info(f"Confidence Threshold: {args.confidence:.0%}")
    logger.info(f"Log File: {args.log_file}")
    
    if args.symbols:
        logger.info(f"Monitoring Pairs: {', '.join(args.symbols)}")
    else:
        logger.info("Monitoring: All configured pairs")
    
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Initialize engine
        engine = InferenceEngine(
            model_dir=args.model_dir,
            confidence_threshold=args.confidence
        )
        
        # Run continuous monitoring
        engine.run_continuous(
            interval_minutes=args.interval,
            symbols=args.symbols
        )
        
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 70)
        logger.info("INFERENCE SERVICE STOPPED BY USER")
        logger.info("=" * 70)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Service crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
