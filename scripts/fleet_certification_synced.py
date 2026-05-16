import sys
import os
import logging
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import InferenceEngine
from core.database import SignalDatabase
from core.performance_gate import get_performance_gate
from data_pipeline.engine import DataEngine
from data_pipeline.labeling import triple_barrier_label

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CERTIFY - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "certification_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Certification")

class MockDataEngine:
    """Mocks DataEngine to return sliced data for historical simulation."""
    def __init__(self, real_engine, current_cutoff):
        self.real_engine = real_engine
        self.current_cutoff = current_cutoff
        
    def fetch(self, symbol, interval="1h", days=60, use_cache=True):
        df = self.real_engine.fetch(symbol, interval=interval, days=days, use_cache=use_cache)
        if df is None: return None
        # Slice up to the simulated 'now'
        return df[df.index <= self.current_cutoff]

    def get_all_pairs(self):
        return self.real_engine.get_all_pairs()

def run_certification():
    logger.info("🚀 Starting 14-Day Fleet Synchronization Audit")
    
    db = SignalDatabase()
    # Ensure DB is clean (User requested reset)
    conn = sqlite3.connect(str(PROJECT_ROOT / "signals.db"))
    conn.execute("DELETE FROM signals")
    conn.commit()
    conn.close()
    logger.info("🗑️ Database purged.")

    real_data_engine = DataEngine()
    symbols = real_data_engine.get_all_pairs()
    
    # Simulation window: Last 14 days
    now_utc = datetime.now(timezone.utc)
    start_sim = now_utc - timedelta(days=14)
    
    # We'll use 1h steps
    # Note: Market might be closed, so we'll only step through actual candle times
    
    for symbol in symbols:
        logger.info(f"--- Auditing {symbol} ---")
        try:
            # 1. Fetch full data for truth labeling
            full_df = real_data_engine.fetch(symbol, interval="1h", days=60)
            if full_df is None or len(full_df) < 100:
                logger.warning(f"Skipping {symbol}: Insufficient data")
                continue
                
            # 2. Generate Truth Labels (Triple Barrier)
            df_labeled = triple_barrier_label(full_df, symbol=symbol)
            
            # 3. Step through the last 14 days
            sim_df = full_df[full_df.index >= start_sim]
            if sim_df.empty:
                logger.warning(f"No data for {symbol} in the last 14 days.")
                continue
                
            engine = InferenceEngine()
            signals_count = 0
            
            # Mock the data engine inside the inference engine
            original_de = engine.data_engine
            
            for timestamp in sim_df.index:
                # Monkey-patch 'now' for this iteration
                engine.data_engine = MockDataEngine(original_de, timestamp)
                
                # Predict (save_to_db=False because we'll handle outcome first)
                # allow_stale=True because we are simulating history
                prediction = engine.predict_symbol(symbol, save_to_db=False, allow_stale=True)
                
                if prediction and not prediction.get('is_locked'):
                    signal_type = prediction.get('signal')
                    if signal_type in ['BUY', 'SELL']:
                        # 4. Determine Outcome from Truth Table
                        truth = df_labeled.loc[timestamp, 'label']
                        outcome = "PENDING"
                        
                        if signal_type == "BUY":
                            if truth == 1: outcome = "SUCCESS"
                            elif truth == 2 or truth == -1: outcome = "FAIL"
                        elif signal_type == "SELL":
                            if truth == 2: outcome = "SUCCESS"
                            elif truth == 1 or truth == -1: outcome = "FAIL"
                        
                        if outcome != "PENDING":
                            # 5. Save to DB with Outcome
                            prediction['outcome'] = outcome
                            prediction['timestamp'] = timestamp.isoformat()
                            db.save_signal(prediction)
                            signals_count += 1
            
            logger.info(f"✅ {symbol}: Generated {signals_count} historical signals.")
            
        except Exception as e:
            logger.error(f"Failed to audit {symbol}: {e}")
            
    # 6. Final Re-Certification
    logger.info("⚖️ Recalculating Performance Matrix...")
    gate = get_performance_gate()
    gate.recompute_from_db(lookback_days=14)
    
    logger.info("✨ Fleet Synchronization Complete.")
    print_whitelist_summary(gate)

def print_whitelist_summary(gate):
    matrix = gate.performance_matrix
    print("\n" + "="*80)
    print(f"{'SYMBOL':<12} | {'DIRECTION':<10} | {'ACCURACY':<10} | {'TRADES':<8} | {'STATUS'}")
    print("-" * 80)
    
    for symbol in sorted(matrix.keys()):
        for direction in ["BUY", "SELL"]:
            tiers = matrix[symbol].get(direction, {})
            # Check highest tier (usually 60 or 70 is enough for certification)
            best_tier = "60"
            if "70" in tiers: best_tier = "70"
            
            data = tiers.get(best_tier)
            if data:
                acc = data.get('accuracy', 0.0)
                trades = data.get('trades', 0)
                status = data.get('status', 'BENCHED')
                print(f"{symbol:<12} | {direction:<10} | {acc:>9.1%} | {trades:>8} | {status}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_certification()
