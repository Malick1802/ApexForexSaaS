#!/usr/bin/env python
# =============================================================================
# Specialist Factory Orchestrator
# =============================================================================
"""
Orchestrates the training of Specialist models for Majors, Minors, and Crosses.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataEngine
from models.specialist_trainer import SpecialistPairTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('specialist_training.log')
    ]
)

logger = logging.getLogger(__name__)


def train_category(category: str, quick: bool = False) -> List[dict]:
    """Train all pairs in a category using Specialist settings."""
    engine = DataEngine()
    pairs = engine.get_all_pairs(category)
    
    if not pairs:
        logger.warning(f"No pairs found for category: {category}")
        return []

    logger.info(f"Starting SPECIALIST training for {len(pairs)} {category.upper()} pairs")
    
    results = []
    for i, symbol in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] Training {symbol}...")
        try:
            # Specialist Config
            trainer = SpecialistPairTrainer(
                symbol,
                epochs=10 if quick else 50,    # Reduced epochs for specialist loop
                n_splits=3 if quick else 5,
                sequence_length=60
            )
            result = trainer.train()
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}", exc_info=True)
            results.append({'symbol': symbol, 'status': 'failed', 'error': str(e)})
            
    return results


def generate_artifact(results: List[dict], category: str) -> str:
    """Generate the artifact summary requested."""
    lines = [
        f"# Specialist Factory Results: {category.upper()}",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Backtest Performance\n",
        "| Pair | Win Rate | Filtered Win Rate | Status |",
        "| :--- | :--- | :--- | :--- |"
    ]
    
    successful = [r for r in results if 'avg_win_rate' in r]
    
    for r in successful:
        symbol = r.get('symbol', 'Unknown')
        wr = r.get('avg_win_rate', 0)
        fwr = r.get('avg_filtered_win_rate', 0)
        
        # Highlight strong performers
        status = "✅ Ready" if fwr > 0.60 else "⚠️ Tuning"
        
        lines.append(f"| **{symbol}** | {wr:.1%} | **{fwr:.1%}** | {status} |")
        
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Specialist Factory')
    parser.add_argument('--category', choices=['majors', 'minors', 'crosses', 'all'], default='all')
    parser.add_argument('--pair', help='Train single pair')
    parser.add_argument('--quick', action='store_true', help='Fast mode for testing')
    
    args = parser.parse_args()
    
    categories = ['majors', 'minors', 'crosses'] if args.category == 'all' else [args.category]
    
    if args.pair:
        logger.info(f"Training single specialist: {args.pair}")
        try:
            trainer = SpecialistPairTrainer(args.pair, epochs=10 if args.quick else 50)
            trainer.train()
            print(f"Training complete for {args.pair}")
        except Exception as e:
            logger.error(f"Error: {e}")
        return

    all_results = []
    
    for cat in categories:
        cat_results = train_category(cat, quick=args.quick)
        
        if cat_results:
            report = generate_artifact(cat_results, cat)
            
            # Save artifact
            artifact_path = f"models/specialist/artifact_{cat}.md"
            os.makedirs("models/specialist", exist_ok=True)
            with open(artifact_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n{report}\n")
            print(f"Artifact saved to {artifact_path}")

if __name__ == "__main__":
    main()
