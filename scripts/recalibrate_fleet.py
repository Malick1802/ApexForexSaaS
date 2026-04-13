import sys
import os
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration import get_calibration_manager

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RECALIBRATE - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Recalibrate")

def main():
    logger.info("🚀 Starting Fleet-Wide Probability Recalibration")
    
    db_path = PROJECT_ROOT / "signals.db"
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}. No history to calibrate.")
        return

    manager = get_calibration_manager()
    
    # Train calibrators from historical closed signals (SUCCESS/FAIL)
    # min_samples=30 ensures we have enough data for a reliable sigmoid fit
    manager.train_from_database(db_path=str(db_path), min_samples=30)
    
    logger.info("✅ Recalibration complete. InferenceEngine will now pick up these new mappings.")

if __name__ == "__main__":
    main()
