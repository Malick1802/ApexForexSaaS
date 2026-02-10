# =============================================================================
# Training Manager - Orchestration Layer
# =============================================================================
"""
Orchestrates training across all currency pairs by category.

Spawns training tasks and generates summary reports.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_pipeline import DataEngine
from models.trainer import PairTrainer


logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TrainingManager:
    """
    Orchestrates LSTM training for all currency pairs.
    
    Features:
    - Category-based training (Majors, Minors, Crosses)
    - Progress tracking
    - Summary report generation
    """
    
    def __init__(
        self,
        base_model_dir: str = "models",
        training_config: Optional[dict] = None
    ):
        """
        Initialize training manager.
        
        Args:
            base_model_dir: Base directory for saving models
            training_config: Optional training configuration
        """
        self.base_model_dir = base_model_dir
        self.training_config = training_config or {
            'sequence_length': 50,
            'n_splits': 5,
            'epochs': 50,
            'batch_size': 32,
            'history_days': 365
        }
        
        self.engine = DataEngine()
        self.results: Dict[str, List[dict]] = {
            'majors': [],
            'minors': [],
            'crosses': []
        }
        
    def train_pair(self, symbol: str, category: str) -> dict:
        """
        Train a single pair.
        
        Args:
            symbol: Currency pair symbol
            category: Category name
            
        Returns:
            Training results
        """
        logger.info(f"[{category.upper()}] Training {symbol}...")
        
        try:
            trainer = PairTrainer(
                symbol=symbol,
                base_model_dir=self.base_model_dir,
                **self.training_config
            )
            result = trainer.train()
            result['category'] = category
            result['status'] = 'success'
            return result
            
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
            return {
                'symbol': symbol,
                'category': category,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_category(self, category: str) -> List[dict]:
        """
        Train all pairs in a category.
        
        Args:
            category: Category name ('majors', 'minors', 'crosses')
            
        Returns:
            List of training results
        """
        pairs = self.engine.get_all_pairs(category)
        logger.info(f"Training {len(pairs)} pairs in {category.upper()}")
        
        results = []
        for i, symbol in enumerate(pairs):
            logger.info(f"[{category.upper()}] {i+1}/{len(pairs)}: {symbol}")
            result = self.train_pair(symbol, category)
            results.append(result)
            self.results[category].append(result)
        
        return results
    
    def train_all(self, categories: Optional[List[str]] = None) -> Dict[str, List[dict]]:
        """
        Train all pairs across all categories.
        
        Args:
            categories: Optional list of categories to train
                       (default: ['majors', 'minors', 'crosses'])
            
        Returns:
            Dict mapping category -> list of results
        """
        categories = categories or ['majors', 'minors', 'crosses']
        
        start_time = datetime.now()
        logger.info(f"Starting training for categories: {categories}")
        
        for category in categories:
            self.train_category(category)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"All training complete in {total_time:.1f} seconds")
        
        return self.results
    
    def train_subset(
        self,
        symbols: List[str],
        parallel: bool = False,
        max_workers: int = 2
    ) -> List[dict]:
        """
        Train a specific subset of pairs.
        
        Args:
            symbols: List of symbols to train
            parallel: Whether to train in parallel
            max_workers: Max parallel workers
            
        Returns:
            List of training results
        """
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.train_pair, s, 'custom'): s 
                    for s in symbols
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for symbol in symbols:
                pair_config = self.engine.get_pair_config(symbol)
                category = pair_config['category'] if pair_config else 'custom'
                results.append(self.train_pair(symbol, category))
        
        return results
    
    def generate_report(self, results: Optional[Dict[str, List[dict]]] = None) -> str:
        """
        Generate markdown report of training results.
        
        Args:
            results: Results dict (uses self.results if not provided)
            
        Returns:
            Markdown formatted report
        """
        results = results or self.results
        
        lines = [
            "# Forex LSTM Training Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n",
        ]
        
        # Summary statistics
        all_results = []
        for cat_results in results.values():
            all_results.extend(cat_results)
        
        successful = [r for r in all_results if r.get('status') == 'success']
        failed = [r for r in all_results if r.get('status') == 'failed']
        
        lines.append("## Summary\n")
        lines.append(f"- **Total Pairs**: {len(all_results)}")
        lines.append(f"- **Successful**: {len(successful)}")
        lines.append(f"- **Failed**: {len(failed)}")
        
        if successful:
            avg_win_rate = sum(r.get('avg_win_rate', 0) for r in successful) / len(successful)
            avg_accuracy = sum(r.get('avg_accuracy', 0) for r in successful) / len(successful)
            lines.append(f"- **Average Win Rate**: {avg_win_rate:.2%}")
            lines.append(f"- **Average Accuracy**: {avg_accuracy:.4f}")
        
        lines.append("\n---\n")
        
        # Results by category
        for category in ['majors', 'minors', 'crosses']:
            cat_results = results.get(category, [])
            if not cat_results:
                continue
            
            lines.append(f"## {category.title()}\n")
            lines.append("| Pair | Win Rate | Accuracy | Samples | Status |")
            lines.append("|------|----------|----------|---------|--------|")
            
            for r in sorted(cat_results, key=lambda x: x.get('avg_win_rate', 0), reverse=True):
                symbol = r.get('symbol', 'Unknown')
                status = r.get('status', 'unknown')
                
                if status == 'success':
                    win_rate = f"{r.get('avg_win_rate', 0):.2%}"
                    accuracy = f"{r.get('avg_accuracy', 0):.4f}"
                    samples = r.get('n_samples', 0)
                    status_icon = "[OK]"
                else:
                    win_rate = "-"
                    accuracy = "-"
                    samples = "-"
                    status_icon = "[FAIL]"
                
                lines.append(f"| {symbol} | {win_rate} | {accuracy} | {samples} | {status_icon} |")
            
            lines.append("\n")
        
        # Top performers
        if successful:
            lines.append("## Top Performers\n")
            top_5 = sorted(successful, key=lambda x: x.get('avg_win_rate', 0), reverse=True)[:5]
            
            for i, r in enumerate(top_5, 1):
                lines.append(
                    f"{i}. **{r['symbol']}** - "
                    f"Win Rate: {r.get('avg_win_rate', 0):.2%}, "
                    f"Accuracy: {r.get('avg_accuracy', 0):.4f}"
                )
            
            lines.append("\n")
        
        # Failed pairs
        if failed:
            lines.append("## Failed Pairs\n")
            for r in failed:
                lines.append(f"- **{r.get('symbol', 'Unknown')}**: {r.get('error', 'Unknown error')}")
            lines.append("\n")
        
        return "\n".join(lines)
    
    def save_report(
        self,
        filepath: Optional[str] = None,
        results: Optional[Dict[str, List[dict]]] = None
    ) -> str:
        """
        Save training report to file.
        
        Args:
            filepath: Output path (default: models/training_report.md)
            results: Results to include
            
        Returns:
            Path to saved report
        """
        filepath = filepath or os.path.join(self.base_model_dir, 'training_report.md')
        
        report = self.generate_report(results)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved report to {filepath}")
        return filepath
    
    def save_results_json(
        self,
        filepath: Optional[str] = None,
        results: Optional[Dict[str, List[dict]]] = None
    ) -> str:
        """
        Save results as JSON for programmatic access.
        
        Args:
            filepath: Output path
            results: Results to save
            
        Returns:
            Path to saved file
        """
        filepath = filepath or os.path.join(self.base_model_dir, 'training_results.json')
        results = results or self.results
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {filepath}")
        return filepath


def run_training(
    categories: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    save_report: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run training.
    
    Args:
        categories: Categories to train (default: all)
        symbols: Specific symbols to train (overrides categories)
        save_report: Whether to save report
        
    Returns:
        Training results
    """
    manager = TrainingManager()
    
    if symbols:
        results = {'custom': manager.train_subset(symbols)}
    else:
        results = manager.train_all(categories)
    
    if save_report:
        manager.save_report()
        manager.save_results_json()
    
    print(manager.generate_report(results))
    
    return results


if __name__ == "__main__":
    # Run training for all pairs
    run_training()
