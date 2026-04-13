import os
import sys
import psutil
import logging
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.core.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MEMORY_TEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoryTest")

def run_stress_test(cycles=2):
    """
    Simulates multiple 31-pair scans and monitors memory.
    If LRU is working, memory should plateau after ~10 models.
    """
    engine = InferenceEngine()
    process = psutil.Process(os.getpid())
    
    initial_mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"🚀 Starting Stress Test. Initial Memory: {initial_mem:.1f} MB")
    
    symbols = engine.data_engine.get_all_pairs()
    logger.info(f"Scan will cover {len(symbols)} symbols ({cycles} cycles)")
    
    for cycle in range(1, cycles + 1):
        logger.info(f"--- Cycle {cycle}/{cycles} Starting ---")
        
        for i, symbol in enumerate(symbols):
            try:
                # Run prediction (loads model if not in cache)
                engine.predict_symbol(symbol, save_to_db=False, allow_stale=True)
                
                curr_mem = process.memory_info().rss / (1024 * 1024)
                cache_size = len(engine._model_cache)
                logger.info(f"[{i+1}/{len(symbols)}] {symbol} | Cache: {cache_size}/10 | RAM: {curr_mem:.1f} MB")
                
                # We expect cache to plateau at 10
                if cache_size > 10:
                    logger.error(f"❌ CACHE OVERFLOW: Size is {cache_size}, expected max 10!")
                    return False
                    
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")
                
        cycle_mem = process.memory_info().rss / (1024 * 1024)
        logger.info(f"✅ Cycle {cycle} Complete. Memory: {cycle_mem:.1f} MB")

    final_mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"🏆 Final Memory: {final_mem:.1f} MB (Delta: {final_mem - initial_mem:.1f} MB)")
    return True

if __name__ == "__main__":
    success = run_stress_test(cycles=1) # 1 full scan for verification
    if success:
        logger.info("🎬 Stress Test PASSED.")
        sys.exit(0)
    else:
        logger.error("🎬 Stress Test FAILED.")
        sys.exit(1)
