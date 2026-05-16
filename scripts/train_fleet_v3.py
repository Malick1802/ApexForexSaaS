"""
Fleet Specialist Factory
========================
Iterates through all symbols and runs the SpecialistV3Trainer.
"""

import sys, subprocess, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Full pair list from v3 config
ALL_PAIRS = [
    "GOLD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "GBPJPY", "EURJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "GBPCHF",
    "EURGBP", "AUDNZD", "NZDCHF", "NZDCAD", "CADCHF", "AUDCHF", "EURCAD",
    "GBPNZD", "EURNZD", "GBPCAD", "USDSGD", "EURAUD", "EURCHF", "GBPAUD",
    "AUDCAD", "EURUSD"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FleetFactory")

def train_fleet():
    logger.info("="*60)
    logger.info("STARTING FLEET SPECIALIST FACTORY")
    logger.info(f"Targets: {len(ALL_PAIRS)} pairs")
    logger.info("="*60)
    
    for symbol in ALL_PAIRS:
        logger.info(f"🚀 Launching Specialist Adaptation for {symbol}...")
        try:
            # We run as a subprocess to ensure memory is fully cleared between pairs
            cmd = [sys.executable, "models/specialist_v3_trainer.py", symbol]
            process = subprocess.run(cmd, check=True)
            if process.returncode == 0:
                logger.info(f"✅ {symbol} SUCCESS")
            else:
                logger.error(f"❌ {symbol} FAILED with return code {process.returncode}")
        except Exception as e:
            logger.error(f"❌ {symbol} CRASHED: {e}")
            
    logger.info("="*60)
    logger.info("FLEET FACTORY COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    train_fleet()
