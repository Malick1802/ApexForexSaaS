# =============================================================================
# Executive - Production Background Worker
# =============================================================================
"""
Autonomous signal generation engine that:
- Polls TwelveData API every 15 minutes
- Respects free tier limit (8 requests/minute)
- Uses specialist models for prediction
- Sends Telegram alerts for high-confidence signals (>85%)
- Logs all activity to system.log
"""

import os
import sys
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from queue import Queue
from threading import Thread, Lock

import numpy as np
import pandas as pd
import yaml
import requests
from telegram import Bot
from telegram.error import TelegramError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from core.database import SignalDatabase
from core.inference import InferenceEngine
from core.notifications import NotificationManager
from tensorflow import keras
import joblib


# =============================================================================
# Setup Logging
# =============================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# Rate limiting is now handled centrally in data_pipeline/providers/twelvedata_provider.py


# =============================================================================
# =============================================================================
# Executive Engine
# =============================================================================

class ExecutiveEngine:
    """
    Production background worker for autonomous signal generation.
    
    Features:
    - 15-minute scanning interval
    - TwelveData rate limiting (via InferenceEngine's DataEngine)
    - High-Confidence "Apex" signals (Default)
    - Telegram alerts
    - Full activity logging
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        target_win_rate: str = "90%",  # Default to high quality (90%+)
        scan_interval_minutes: int = 15
    ):
        logger.info("="*70)
        logger.info("EXECUTIVE ENGINE - STARTING")
        logger.info("="*70)
        
        self.target_win_rate = target_win_rate
        self.scan_interval_minutes = scan_interval_minutes
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        # cooldown_minutes=0 so we can scan every interval without internal blocking
        self.inference_engine = InferenceEngine(config_path=config_path, cooldown_minutes=0)
        self.db = SignalDatabase()
        self.notifier = NotificationManager()
        
        # Recent signals tracker (deduplication)
        self._recent_signals: Dict[str, datetime] = {}
        self._cooldown_minutes = 60
        
        logger.info(f"Target Win Rate: {target_win_rate}")
        logger.info(f"Scan Interval: {scan_interval_minutes} minutes")
        logger.info(f"Telegram Alerts: {'Enabled' if self.notifier.enabled else 'Disabled'}")
        logger.info("="*70)

    def _is_duplicate_signal(self, symbol: str, last_candle_time: pd.Timestamp) -> bool:
        """
        Check if we already generated a signal for this specific candle.
        Prevents spamming signals when market is closed (data is static).
        """
        try:
            # Check DB for last signal
            recent = self.db.get_recent_signals(limit=1, symbol=symbol)
            if not recent:
                return False
                
            last_signal_ts_str = recent[0]['timestamp']
            last_signal_ts = pd.to_datetime(last_signal_ts_str)
            
            # Ensure timezone awareness compatibility
            if last_candle_time.tzinfo is None:
                last_candle_time = last_candle_time.tz_localize('UTC')
            if last_signal_ts.tzinfo is None:
                last_signal_ts = last_signal_ts.tz_localize('UTC')
                
            # If the last signal was generated AFTER the candle closed, we already saw this data.
            # Adding a small buffer (e.g., 1 minute) to allow for processing time differences
            if last_signal_ts > last_candle_time:
                return True
                
            return False
        except Exception as e:
            logger.warning(f"Deduplication check failed: {e}")
            return False

    def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single symbol and generate signal if criteria met.
        Only saves a new BUY/SELL signal if there's no existing active signal for this pair.
        Returns signal dict or None.
        """
        try:
            # Use InferenceEngine for prediction
            # save_to_db=False because we handle DB saving here after deduplication
            result = self.inference_engine.predict_symbol(
                symbol,
                save_to_db=False,
                win_rate=self.target_win_rate
            )
            
            if not result:
                return None
            
            signal = result['signal']
            
            if signal in ('BUY', 'SELL'):
                # DEDUP GUARD: Only save if no active signal exists for this pair
                if self.db.has_active_signal(symbol):
                    logger.debug(f"⏭ {symbol}: Active signal already exists, skipping duplicate")
                    return None
                
                # New actionable signal — save it
                self.db.save_signal(result)
                
                log_icon = "🟢" if signal == "BUY" else "🔴"
                logger.info(
                    f"{log_icon} NEW SIGNAL: {symbol} {signal} @ {result['price_at_signal']:.5f} "
                    f"(Conf: {result['confidence']:.1%})"
                )
                
                # Send Telegram alert
                self.notifier.send_signal_alert(result)
                
                # Update cooldown
                self._recent_signals[symbol] = datetime.now()
                return result
            else:
                # WAIT signals: save with N/A outcome (for dashboard timestamp tracking)
                result['outcome'] = 'N/A'
                self.db.save_signal(result)
            
            return None

            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            return None
    
    def run_scan(self, symbols: List[str]):
        """Execute a single market scan across all symbols."""
        start_time = time.time()
        logger.info(f"--- MARKET SCAN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        logger.info(f"Scanning {len(symbols)} pairs for {self.target_win_rate} setups...")
        
        # Expire stale signals (>48h old) before scanning
        self.db.expire_stale_signals(max_age_hours=48)
        
        signals_generated = 0
        
        for i, symbol in enumerate(symbols):
            result = self.analyze_symbol(symbol)
            if result:
                signals_generated += 1
            
            # Stagger requests to avoid bursting the 8/min limit
            # 7.5s * 8 = 60s. This keeps us strictly within limits.
            if i < len(symbols) - 1:
                time.sleep(7.5)
        
        elapsed = time.time() - start_time
        logger.info(f"--- SCAN COMPLETE: {signals_generated} new signals in {elapsed:.1f}s ---")
        
        # Monitor outcomes after each scan
        self.monitor_active_signals()
        logger.info("")
    
    def monitor_active_signals(self):
        """Check all ACTIVE signals against current prices to determine outcomes (SUCCESS/FAIL)."""
        active_signals = self.db.get_active_signals()
        if not active_signals:
            return
            
        logger.info(f"Watchdog: Checking {len(active_signals)} active signals for outcomes...")
        
        for sig in active_signals:
            symbol = sig['symbol']
            try:
                # Fetch latest price
                df = self.inference_engine.data_engine.fetch(symbol, interval="1m", days=1, use_cache=False)
                if df.empty:
                    continue
                    
                current_price = df['close'].iloc[-1]
                tp = sig['tp_price']
                sl = sig['sl_price']
                direction = sig['signal']
                
                # Check outcome
                if direction == 'BUY':
                    if current_price >= tp:
                        self.db.update_signal_outcome(sig['id'], 'SUCCESS')
                        logger.info(f"🎯 SUCCESS: {symbol} hit TP at {current_price}")
                    elif current_price <= sl:
                        self.db.update_signal_outcome(sig['id'], 'FAIL')
                        logger.info(f"❌ FAIL: {symbol} hit SL at {current_price}")
                elif direction == 'SELL':
                    if current_price <= tp:
                        self.db.update_signal_outcome(sig['id'], 'SUCCESS')
                        logger.info(f"🎯 SUCCESS: {symbol} hit TP at {current_price}")
                    elif current_price >= sl:
                        self.db.update_signal_outcome(sig['id'], 'FAIL')
                        logger.info(f"❌ FAIL: {symbol} hit SL at {current_price}")
                        
            except Exception as e:
                logger.error(f"Watchdog failed for {symbol}: {e}")
    
    def run_continuous(self, symbols: Optional[List[str]] = None):
        """
        Run continuous market monitoring.
        
        Args:
            symbols: List of symbols to monitor (None = all configured)
        """
        if symbols is None:
            # Use InferenceEngine to get symbols if possible, or fallback
            # InferenceEngine doesn't have get_all_pairs exposed directly maybe?
            # It has self.data_engine.
            symbols = self.inference_engine.data_engine.get_all_pairs()
        
        logger.info(f"Starting continuous monitoring: {len(symbols)} pairs")
        logger.info(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        logger.info(f"Target Win Rate: {self.target_win_rate}")
        logger.info("")
        
        try:
            while True:
                self.run_scan(symbols)
                
                # Sleep until next scan
                sleep_time = self.scan_interval_minutes * 60
                next_scan = datetime.now() + timedelta(seconds=sleep_time)
                logger.info(f"Next scan: {next_scan.strftime('%H:%M:%S')} ({self.scan_interval_minutes} min)")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("="*70)
            logger.info("EXECUTIVE ENGINE STOPPED BY USER")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Executive engine crashed: {e}", exc_info=True)
            raise


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Executive Engine - Production Background Worker")
    parser.add_argument('--win-rate', type=str, default='90%', help='Target Win Rate (e.g., 90%, Apex)')
    parser.add_argument('--interval', type=int, default=15, help='Scan interval in minutes (default: 15)')
    parser.add_argument('--symbols', nargs='+', default=None, help='Specific symbols to monitor')
    
    args = parser.parse_args()
    
    # Initialize and run
    engine = ExecutiveEngine(
        target_win_rate=args.win_rate,
        scan_interval_minutes=args.interval
    )
    
    engine.run_continuous(symbols=args.symbols)

