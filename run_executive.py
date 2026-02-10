"""
Executive Engine Runner - Production Background Worker
=======================================================

This script runs the executive engine that automatically generates
trading signals and sends Telegram alerts.

Usage:
    python run_executive.py
    python run_executive.py --confidence 0.85 --interval 15
    python run_executive.py --symbols EURUSD GBPUSD USDJPY
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from core.executive import ExecutiveEngine
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Executive Engine - Automatic Signal Generation with Telegram Alerts"
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.85,
        help='Minimum confidence threshold for signals (default: 0.85 = 85%%)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=15,
        help='Minutes between market scans (default: 15)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Specific symbols to monitor (default: all configured pairs)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXECUTIVE ENGINE - PRODUCTION WORKER")
    print("="*70)
    print(f"Confidence Threshold: {args.confidence:.0%}")
    print(f"Scan Interval: {args.interval} minutes")
    print(f"Rate Limit: 8 requests/minute (TwelveData Free Tier)")
    print(f"Telegram Alerts: Enabled (if configured)")
    print(f"Logging: logs/system.log")
    print("="*70)
    print("")
    
    # Initialize engine
    engine = ExecutiveEngine(
        confidence_threshold=args.confidence,
        scan_interval_minutes=args.interval
    )
    
    # Run continuous monitoring
    engine.run_continuous(symbols=args.symbols)


if __name__ == "__main__":
    main()
