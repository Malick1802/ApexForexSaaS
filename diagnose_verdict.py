import logging
import sys
import os
from pathlib import Path

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Diagnostics")

def run_diagnostics():
    logger.info("Starting ApexForex Diagnostics...")
    
    # 1. Path Check
    root = Path(__file__).parent.resolve()
    logger.info(f"Root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        logger.info("Added root to sys.path")

    # 2. Try imports
    try:
        from core.inference import InferenceEngine
        logger.info("✅ core.inference imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import core.inference: {e}")
        return

    # 3. Initialize Engine
    try:
        engine = InferenceEngine()
        logger.info("✅ InferenceEngine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize InferenceEngine: {e}")
        return

    # 4. Check Models
    symbol = "EURUSD"
    logger.info(f"Testing prediction for {symbol}...")
    
    try:
        # We'll peak into the loading logic
        expert = engine.load_phase3_expert(symbol)
        logger.info(f"Expert Model found: {expert is not None}")
        
        foundation = engine.load_foundation_model(symbol)
        logger.info(f"Foundation Model found: {foundation is not None}")
        
        specialist = engine.load_models(symbol)
        logger.info(f"Specialist Model found: {specialist is not None}")
        
        # 5. Run Prediction
        result = engine.predict_symbol(symbol, save_to_db=False, allow_stale=True)
        if result:
            logger.info(f"✅ Prediction Successful! Signal: {result['signal']}, Confidence: {result['confidence']:.2%}")
            logger.info(f"Model Version: {result.get('model_version')}")
        else:
            logger.error(f"❌ Prediction returned NONE for {symbol}")
            
    except Exception as e:
        logger.error(f"❌ Prediction crashed: {e}", exc_info=True)

if __name__ == "__main__":
    run_diagnostics()
