import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.engine import DataEngine
from core.core.database import SignalDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonteCarloAudit")

class MonteCarloStressTester:
    def __init__(self):
        self.db = SignalDatabase()
        self.engine = DataEngine()
        
    def run_permutation_test(self, symbol="EURUSD", num_sims=1000):
        """
        Shuffles the order of historical trades to see if profitability 
        is dependent on a lucky sequence of events.
        """
        logger.info(f"🎲 Running Monte Carlo Permutation Test (n={num_sims}) for {symbol}...")
        
        # Load signals from DB or a historical log
        signals = self.db.get_recent_signals(limit=500)
        resolved = [s for s in signals if s['symbol'] == symbol and s['status'] in ['SUCCESS', 'FAIL']]
        
        if not resolved:
            logger.warning(f"No resolved trades found for {symbol} in DB. Using simulated data for stress test demo.")
            # Simulated: 55% win rate, 1:1.5 RR, 100 trades
            returns = [1.5 if i < 55 else -1.0 for i in range(100)]
        else:
            returns = [1.5 if s['status'] == 'SUCCESS' else -1.0 for s in resolved]
            
        original_equity = np.cumsum(returns)
        final_profit = original_equity[-1]
        
        sim_results = []
        for _ in range(num_sims):
            shuffled = returns.copy()
            random.shuffle(shuffled)
            equity_curve = np.cumsum(shuffled)
            max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve)
            sim_results.append({
                'final_profit': equity_curve[-1],
                'max_dd': max_drawdown
            })
            
        avg_dd = np.mean([r['max_dd'] for r in sim_results])
        worst_dd = np.max([r['max_dd'] for r in sim_results])
        failure_prob = np.mean([1 if r['final_profit'] <= 0 else 0 for r in sim_results])
        
        logger.info(f"📈 Original Final Profit: {final_profit:.2f}R")
        logger.info(f"🛡️ Monte Carlo Failure Probability: {failure_prob:.1%}")
        logger.info(f"📉 Average Shuffled Drawdown: {avg_dd:.2f}R")
        logger.info(f"💀 Worst-Case Shuffled Drawdown: {worst_dd:.2f}R")
        
        return {
            'symbol': symbol,
            'failure_prob': failure_prob,
            'avg_drawdown': avg_dd,
            'original_profit': final_profit
        }

    def run_noise_sensitivity_test(self, symbol="EURUSD", noise_pips=0.5):
        """
        Injects Gaussian noise into price data and re-runs inference 
        to see if accuracy collapses (checks for overfitting).
        """
        logger.info(f"🔊 Injecting {noise_pips} pips of noise into {symbol} data...")
        
        from core.core.inference import InferenceEngine
        inf_engine = InferenceEngine()
        
        # Fetch 200 bars for testing
        df = self.engine.fetch(symbol, interval="1h", days=30)
        if df is None or len(df) < 60: return None
        
        # Original Inference
        res_orig = inf_engine.predict_symbol(symbol, df)
        if not res_orig:
            logger.error("Original inference failed. Skipping noise test.")
            return False
            
        orig_sig = res_orig['signal']
        orig_conf = res_orig['confidence']
        
        # Inject Noise
        pip_val = self.engine.get_pip_value(symbol)
        df_noise = df.copy()
        noise = np.random.normal(0, noise_pips * pip_val, size=len(df))
        df_noise['close'] += noise
        df_noise['high'] += np.abs(noise)
        df_noise['low'] -= np.abs(noise)
        
        # Noisy Inference
        res_noise = inf_engine.predict_symbol(symbol, df_noise)
        if not res_noise:
            logger.error("Noisy inference failed.")
            return False
            
        noise_sig = res_noise['signal']
        noise_conf = res_noise['confidence']
        
        conf_diff = abs(orig_conf - noise_conf)
        is_robust = orig_sig == noise_sig
        
        logger.info(f"🧠 Original: {orig_sig} ({orig_conf:.1%}) | Noisy: {noise_sig} ({noise_conf:.1%})")
        logger.info(f"🛡️ Robustness Check: {'PASSED' if is_robust else 'FAILED'} (Delta: {conf_diff:.2%})")
        
        return is_robust

if __name__ == "__main__":
    tester = MonteCarloStressTester()
    tester.run_permutation_test("EURUSD")
    tester.run_noise_sensitivity_test("EURUSD")
