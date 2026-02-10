#!/usr/bin/env python
# =============================================================================
# Enhanced LSTM Training Script (60%+ Win Rate Target)
# =============================================================================
"""
Train enhanced LSTM models with confidence filtering for higher win rates.

Usage:
    python train_enhanced.py --symbol EURUSD        # Train single pair
    python train_enhanced.py --category majors      # Train category
    python train_enhanced.py --binary EURUSD        # Train binary classifiers
"""

import argparse
import logging
import sys
import os

# Force CPU to avoid GPU hangs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataEngine
from models.enhanced_trainer import EnhancedPairTrainer, BinaryPairTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_training.log')
    ]
)

logger = logging.getLogger(__name__)


def train_enhanced_pair(symbol: str, quick: bool = False) -> dict:
    """Train a single pair with enhanced settings."""
    config = {
        'sequence_length': 40 if quick else 60,
        'n_splits': 3 if quick else 5,
        'epochs': 30 if quick else 100,
        'batch_size': 32,
        'history_days': 180 if quick else 365,
        'confidence_threshold': 0.7,
        'use_class_weights': True
    }
    
    trainer = EnhancedPairTrainer(symbol, **config)
    return trainer.train()


def train_binary_pair(symbol: str, quick: bool = False) -> Dict[str, dict]:
    """Train binary BUY/SELL classifiers for a pair."""
    results = {}
    
    epochs = 10 if quick else 50
    
    for signal_type in ['BUY', 'SELL']:
        trainer = BinaryPairTrainer(
            symbol=symbol,
            signal_type=signal_type,
            epochs=epochs
        )
        results[signal_type] = trainer.train()
    
    return results


def train_category(category: str, quick: bool = False) -> List[dict]:
    """Train all pairs in a category."""
    engine = DataEngine()
    pairs = engine.get_all_pairs(category)
    
    logger.info(f"Training {len(pairs)} pairs in {category.upper()} (enhanced mode)")
    
    results = []
    for i, symbol in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] Training {symbol}...")
        try:
            result = train_enhanced_pair(symbol, quick=quick)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
            results.append({'symbol': symbol, 'status': 'failed', 'error': str(e)})
    
    return results


def generate_report(results: List[dict]) -> str:
    """Generate markdown report for enhanced training."""
    lines = [
        "# Enhanced LSTM Training Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model Type**: Enhanced LSTM with Attention + Confidence Filtering",
        "\n---\n",
    ]
    
    successful = [r for r in results if 'avg_filtered_win_rate' in r]
    
    if successful:
        avg_standard = sum(r.get('avg_win_rate', 0) for r in successful) / len(successful)
        avg_filtered = sum(r.get('avg_filtered_win_rate', 0) for r in successful) / len(successful)
        
        lines.append("## Summary\n")
        lines.append(f"- **Total Pairs Trained**: {len(results)}")
        lines.append(f"- **Successful**: {len(successful)}")
        lines.append(f"- **Standard Win Rate**: {avg_standard:.2%}")
        lines.append(f"- **FILTERED Win Rate (>70% confidence)**: {avg_filtered:.2%}")
        lines.append("\n---\n")
        
        lines.append("## Results by Pair\n")
        lines.append("| Pair | Standard Win Rate | Filtered Win Rate | Signal Rate | Samples |")
        lines.append("|------|-------------------|-------------------|-------------|---------|")
        
        for r in sorted(successful, key=lambda x: x.get('avg_filtered_win_rate', 0), reverse=True):
            symbol = r.get('symbol', 'Unknown')
            std_wr = r.get('avg_win_rate', 0)
            flt_wr = r.get('avg_filtered_win_rate', 0)
            sig_rate = r.get('final_signal_rate', 0)
            samples = r.get('n_samples', 0)
            
            # Highlight pairs with 60%+ filtered win rate
            marker = " [TARGET]" if flt_wr >= 0.60 else ""
            
            lines.append(
                f"| {symbol}{marker} | {std_wr:.2%} | **{flt_wr:.2%}** | {sig_rate:.1%} | {samples} |"
            )
        
        lines.append("\n")
        
        # Pairs that hit 60% target
        target_pairs = [r for r in successful if r.get('avg_filtered_win_rate', 0) >= 0.60]
        if target_pairs:
            lines.append("## Pairs Meeting 60% Target\n")
            for r in target_pairs:
                lines.append(f"- **{r['symbol']}**: {r['avg_filtered_win_rate']:.2%} filtered win rate")
            lines.append("\n")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Enhanced LSTM Training')
    parser.add_argument('--symbol', help='Train specific symbol')
    parser.add_argument('--category', choices=['majors', 'minors', 'crosses'])
    parser.add_argument('--binary', help='Train binary classifiers for symbol')
    parser.add_argument('--quick', action='store_true', help='Quick training mode')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    try:
        if args.binary:
            logger.info(f"Training binary classifiers for {args.binary}")
            results = train_binary_pair(args.binary, quick=args.quick)
            print(f"\nBinary training complete for {args.binary}:")
            for signal, metrics in results.items():
                print(f"  {signal}: acc={metrics['val_accuracy']:.2%}, "
                      f"high_conf={metrics['high_confidence_accuracy']:.2%}")
        
        elif args.symbol:
            logger.info(f"Training enhanced model for {args.symbol}")
            result = train_enhanced_pair(args.symbol, quick=args.quick)
            print(f"\n{args.symbol} Results:")
            print(f"  Standard Win Rate: {result['avg_win_rate']:.2%}")
            print(f"  FILTERED Win Rate: {result['avg_filtered_win_rate']:.2%}")
        
        elif args.category:
            results = train_category(args.category, quick=args.quick)
            report = generate_report(results)
            print("\n" + "="*80)
            print(report)
            print("="*80)
            
            # Save report
            report_path = f"models/enhanced/training_report_{args.category}.md"
            os.makedirs("models/enhanced", exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
        
        else:
            parser.print_help()
            return 1
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {duration:.1f} seconds")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
