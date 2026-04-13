"""
Progressive Specialist Training (Parallel Workers)
====================================================
Trains separate BUY/SELL binary classifiers for each currency pair.
Supports parallel workers for faster fleet training.

Usage:
    # Single worker (all pairs):
    python scripts/train_specialists_progressive.py --target 90

    # Parallel workers (split fleet):
    python scripts/train_specialists_progressive.py --target 90 --worker 1 --total-workers 3
    python scripts/train_specialists_progressive.py --target 90 --worker 2 --total-workers 3
    python scripts/train_specialists_progressive.py --target 90 --worker 3 --total-workers 3
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SPECIALIST - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "specialist_progressive.log")
    ]
)
logger = logging.getLogger("SpecialistProgressive")


def get_all_pairs():
    """Load all currency pairs from config.yaml."""
    import yaml
    with open(PROJECT_ROOT / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pairs = []
    for category in ['majors', 'minors', 'crosses']:
        pair_list = config.get('currency_pairs', {}).get(category, [])
        pairs.extend([p['symbol'] for p in pair_list])
    
    return sorted(list(set(pairs)))


def run_training(target_accuracy: int = 90, pair: str = None, 
                 worker: int = None, total_workers: int = None):
    """Run specialist training at a given accuracy target."""
    from models.specialist_factory import SpecialistFactory
    
    min_wr = target_accuracy / 100.0
    
    # Save to specialist directory
    base_dir = str(PROJECT_ROOT / "models" / "specialist")
    
    factory = SpecialistFactory(
        base_dir=base_dir,
        min_win_rate=min_wr,
        min_samples=1000,
        provider_name="mt5"
    )
    
    start_time = time.time()
    
    if pair:
        # Single pair training
        logger.info(f"Training single pair: {pair}")
        buy_ok = factory.train_specialist(pair.upper(), "BUY")
        sell_ok = factory.train_specialist(pair.upper(), "SELL")
        factory._update_fleet_report(1, 1, pair.upper())
    elif worker is not None and total_workers is not None:
        # Parallel worker mode
        all_pairs = get_all_pairs()
        
        # Split pairs across workers
        worker_pairs = [p for i, p in enumerate(all_pairs) if i % total_workers == (worker - 1)]
        
        logger.info("=" * 60)
        logger.info(f"WORKER {worker}/{total_workers} — Target: {target_accuracy}%")
        logger.info(f"Assigned pairs ({len(worker_pairs)}): {', '.join(worker_pairs)}")
        logger.info("=" * 60)
        
        # Pass 1: Unattempted only
        logger.info("=== PASS 1: Training First-Time Pairs ===")
        for idx, symbol in enumerate(worker_pairs):
            buy_conf = Path(base_dir) / symbol / "BUY" / "config.json"
            sell_conf = Path(base_dir) / symbol / "SELL" / "config.json"
            
            if not buy_conf.exists():
                buy_ok = factory.train_specialist(symbol, "BUY")
            else:
                logger.info(f"⏭️ Skipping {symbol} BUY (Already got first pass)")
                
            if not sell_conf.exists():
                sell_ok = factory.train_specialist(symbol, "SELL")
            else:
                logger.info(f"⏭️ Skipping {symbol} SELL (Already got first pass)")
                
            factory._update_fleet_report(idx + 1, len(worker_pairs), symbol)
            
        # Pass 2: Continuous re-evaluation of non-certified pairs
        logger.info("=== PASS 2: Re-evaluating Non-Certified Pairs ===")
        while True:
            for idx, symbol in enumerate(worker_pairs):
                # Factory automatically skips fully certified ones inside this method
                buy_ok = factory.train_specialist(symbol, "BUY")
                sell_ok = factory.train_specialist(symbol, "SELL")
                factory._update_fleet_report(idx + 1, len(worker_pairs), symbol)
                logger.info(f"Worker {worker} (Pass 2): Completed {symbol} ({idx+1}/{len(worker_pairs)})")
            logger.info("Finished a full re-evaluation loop. Starting over...")
    else:
        # Full fleet (sequential)
        factory.train_fleet()
        
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE in {elapsed/60:.1f} minutes")
    logger.info("=" * 60)
    
    # Generate final summary report
    generate_report(base_dir, target_accuracy)


def generate_report(base_dir: str, target: int):
    """Generate a summary report of all trained specialist models."""
    base = Path(base_dir)
    
    results = []
    for pair_dir in sorted(base.iterdir()):
        if not pair_dir.is_dir():
            continue
        symbol = pair_dir.name
        for signal_type in ["BUY", "SELL"]:
            config_path = pair_dir / signal_type / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    data = json.load(f)
                wr = data.get("win_rate", 0)
                trades = data.get("trades", 0)
                results.append({
                    "symbol": symbol,
                    "type": signal_type,
                    "win_rate": wr,
                    "trades": trades,
                    "golden": wr >= target / 100.0 and trades >= 1000
                })
            else:
                results.append({
                    "symbol": symbol,
                    "type": signal_type,
                    "win_rate": 0,
                    "trades": 0,
                    "golden": False
                })
    
    if not results:
        logger.info("No models found.")
        return
    
    # Print table
    print(f"\n{'='*65}")
    print(f"  GOLDEN SIGNAL REPORT — Target: {target}% + 1000 trades")
    print(f"{'='*65}")
    print(f"{'Pair':<10} {'Type':<6} {'Win Rate':>10} {'Trades':>8} {'Status':>10}")
    print("-" * 65)
    
    golden_count = 0
    for r in results:
        status = "🏆 GOLDEN" if r["golden"] else "❌ FAIL"
        if r["golden"]:
            golden_count += 1
        print(f"{r['symbol']:<10} {r['type']:<6} {r['win_rate']:>9.1%} {r['trades']:>8} {status:>10}")
    
    print("-" * 65)
    print(f"Golden Certified: {golden_count}/{len(results)}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Progressive Specialist Training')
    parser.add_argument('--target', type=int, default=90, choices=[60, 70, 80, 90, 95],
                        help='Target accuracy percentage (default: 90)')
    parser.add_argument('--pair', type=str, default=None,
                        help='Train single pair (e.g., EURUSD)')
    parser.add_argument('--worker', type=int, default=None,
                        help='Worker number (1-based) for parallel training')
    parser.add_argument('--total-workers', type=int, default=None,
                        help='Total number of parallel workers')
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    
    run_training(
        target_accuracy=args.target,
        pair=args.pair,
        worker=args.worker,
        total_workers=args.total_workers
    )
