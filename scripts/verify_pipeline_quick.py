import os
import sys
import logging
from datetime import datetime

# Inject project root directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger('PipelineVerifier')

def run_checks():
    logger.info("Starting Comprehensive Pipeline Verification...")

    # Stage 1: Data Fetching
    logger.info("--- Stage 1: Data Engine ---")
    try:
        from data_pipeline.engine import DataEngine
        engine = DataEngine()
        data = engine.fetch_data('EURUSD', '1h', limit=50)
        if data is not None and not data.empty:
            logger.info(f"[SUCCESS] Data Engine fetched EURUSD (Rows: {len(data)})")
        else:
            logger.error("[FAIL] Data Engine returned empty/None data")
    except Exception as e:
        logger.error(f"[FAIL] Data Engine Exception: {e}")

    # Stage 2: Models
    logger.info("--- Stage 2: Trained Models ---")
    try:
        import glob
        models = glob.glob(os.path.join(project_root, 'models/trained_models/*h5')) + glob.glob(os.path.join(project_root, 'models/trained_models/*keras'))
        if len(models) > 0:
            logger.info(f"[SUCCESS] Found {len(models)} trained model(s) on disk.")
            
            from core.core.inference import InferenceEngine
            inf = InferenceEngine()
            # Suppress excessive info logs for predict
            logging.getLogger('core.inference').setLevel(logging.WARNING)
            pred = inf.predict_symbol('EURUSD')
            if pred and 'signal' in pred:
                logger.info(f"[SUCCESS] Inference Engine successfully generated prediction: {pred['signal']} for EURUSD (Conf: {pred.get('confidence',0):.1%})")
            else:
                logger.error("[FAIL] Inference Engine failed to output standard response dict.")
        else:
            logger.error("[FAIL] No trained models found in models/trained_models/!")
    except Exception as e:
        logger.error(f"[FAIL] Trained Models Exception: {e}", exc_info=True)

    # Stage 3: Executive Engine & Certification
    logger.info("--- Stage 3: Executive Engine & Whitelist ---")
    try:
        from core.core.executive import ExecutiveEngine
        from core.core.performance_gate import PerformanceGate
        gate = PerformanceGate()
        status_eurusd = gate.get_tier_status('EURUSD', 0.6)
        logger.info(f"[SUCCESS] Performance Gate read matrix successfully (EURUSD 60% = {status_eurusd})")
        
        exec_engine = ExecutiveEngine()
        logger.info(f"[SUCCESS] Executive Engine initialized (Target Win Rate: {exec_engine.target_win_rate})")
    except Exception as e:
        logger.error(f"[FAIL] Executive Engine Exception: {e}")

    # Stage 4: Notifications (Telegram check)
    logger.info("--- Stage 4: Notification Manager ---")
    try:
        from core.core.notifications import NotificationManager
        notifier = NotificationManager()
        if notifier.enabled and notifier.bot_token:
            logger.info(f"[SUCCESS] Telegram configured correctly. Shadow Trades enabled: {notifier.telegram_config.get('notify_shadow_trades')}")
        else:
            logger.warning("[WARNING] Telegram might not be fully configured.")
    except Exception as e:
        logger.error(f"[FAIL] Notification Exception: {e}")

    # Stage 5: Sentinel Resolution
    logger.info("--- Stage 5: Sentinel (Database Resolution) ---")
    try:
        from core.sentinel import TradingSentinel
        sentinel = TradingSentinel()
        logger.info("[SUCCESS] Sentinel loaded. Ready to monitor active signals via SQLite.")
    except Exception as e:
        logger.error(f"[FAIL] Sentinel Exception: {e}")

    logger.info("Verification Complete.")

if __name__ == '__main__':
    run_checks()
