"""
Full Phase 4 Overnight Retrain
===============================
This script fully retrains the AI on the new Macro Features (VIX + Yield Curve)
and the extended 5-year dataset.

Steps:
  1. Train Foundation TFT Model (Phase 2 constraint)
  2. Map Foundation Weights to 31 Currency Pairs (Phase 3 Adaptation)

This process will take 4-6 hours depending on hardware.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/overnight_retrain.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FullRetrain")

def run_retrain():
    logger.info("="*60)
    logger.info("  STARTING FULL OVERNIGHT RETRAIN")
    logger.info("="*60)
    
    # Environment config
    env = os.environ.copy()
    env['TF_ENABLE_ONEDNN_OPTS'] = '0'
    env['PYTHONPATH'] = os.getcwd()
    
    # 1. Foundation Training 
    logger.info("STEP 1: Training Foundation TFT Model (Global Rules)")
    try:
        subprocess.run([sys.executable, "start_global_training.py"], env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Foundation Training Failed: {e}")
        return False
        
    logger.info("Foundation Training Complete!")
    
    # 2. Specialist Adaptation
    logger.info("\nSTEP 2: Specialist Adaptation (Fine-tuning 31 pairs)")
    try:
        subprocess.run([sys.executable, "start_adaptation.py"], env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Adaptation Training Failed: {e}")
        return False
        
    logger.info("Adaptation Training Complete!")
    
    logger.info("="*60)
    logger.info("  PHASE 4 RETRAIN FULLY COMPLETE")
    logger.info("="*60)
    return True

if __name__ == "__main__":
    run_retrain()
