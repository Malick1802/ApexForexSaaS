import sys
import os
from pathlib import Path
import shutil
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.win_rate_trainer import WinRateFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_robustness_pass():
    factory = WinRateFactory()
    
    # Full Fleet of 31 Pairs
    pairs = [
        # Majors
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        # Minors
        "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "AUDCAD", "AUDJPY", "AUDNZD",
        # Crosses & Exotic
        "CADJPY", "CHFJPY", "NZDJPY", "CADCHF", "NZDCAD", "NZDCHF", "EURNZD", "GBPNZD", "AUDCHF", "EURSGD", "USDSGD", "USDHKD"
    ]
    
    logger.info(f"Starting Robustness Pass retraining for {len(pairs)} pairs...")
    for i, symbol in enumerate(pairs):
        try:
            logger.info(f"[{i+1}/{len(pairs)}] Processing {symbol}...")
            # Force clear existing expert models for this symbol to ensure Phase 6 logic is applied
            symbol_dir = Path("models") / symbol
            if symbol_dir.exists():
                logger.info(f"🗑️ Clearing existing models for {symbol}")
                shutil.rmtree(symbol_dir)
                
            factory.create_expert_models(symbol)
            logger.info(f"✅ Robustness Pass complete for {symbol}")
            
            # Update the global report after each pair
            factory.generate_comprehensive_report()
            
        except Exception as e:
            logger.error(f"❌ Robustness Pass failed for {symbol}: {e}")
            
    logger.info("🏁 Full Fleet Robustness Pass Completed.")

if __name__ == "__main__":
    run_robustness_pass()
