
import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bayesian_engine import get_bayesian_engine
from core.core.performance_gate import get_performance_gate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestBayesian")

def test_bayesian_flow():
    print("--- 🧠 BAYESIAN FLOW TEST ---")
    
    # 1. Test Static Calculation
    engine = get_bayesian_engine()
    
    # High Confidence Case
    alpha_high = 10.0 # 8 wins
    beta_high = 2.0  # 0 losses (init was 2,2)
    scaling_high = engine.calculate_confidence_scaling(alpha_high, beta_high, 0.70)
    print(f"High Performance (8-0): Scaling Factor = {scaling_high:.2f} (Expected > 1.0)")
    
    # Low Confidence Case
    alpha_low = 2.0
    beta_low = 10.0 # 8 losses
    scaling_low = engine.calculate_confidence_scaling(alpha_low, beta_low, 0.70)
    print(f"Low Performance (0-8): Scaling Factor = {scaling_low:.2f} (Expected < 1.0)")
    
    # 2. Test Performance Gate Integration
    gate = get_performance_gate()
    symbol = "EURUSD"
    tier = 70
    
    print(f"\nUdating Bayesian for {symbol}@{tier}...")
    gate.update_bayesian(symbol, tier, "SUCCESS")
    gate.update_bayesian(symbol, tier, "SUCCESS")
    gate.update_bayesian(symbol, tier, "FAIL")
    
    data = gate.performance_matrix[symbol][str(tier)]
    print(f"Result: alpha={data['alpha']}, beta={data['beta']}, status={data['status']}")
    
    # 3. Test Inference Scaling (Simulation)
    raw_conf = 0.90
    final_conf = raw_conf * engine.calculate_confidence_scaling(data['alpha'], data['beta'], tier/100.0)
    print(f"Inference Simulation: Raw={raw_conf:.1%} -> Adjusted={final_conf:.1%}")
    
    print("\n--- ✅ TEST COMPLETE ---")

if __name__ == "__main__":
    test_bayesian_flow()
