import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import SignalDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_signals():
    logger.info("Resetting signals database...")
    db = SignalDatabase()
    success = db.clear_signals()
    
    if success:
        logger.info("✅ Signals cleared successfully.")
    else:
        logger.error("❌ Failed to clear signals.")

if __name__ == "__main__":
    reset_signals()
