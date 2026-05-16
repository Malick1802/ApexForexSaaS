# =============================================================================
# ApexForex SaaS - The Executive
# =============================================================================
"""
The 'Executive' script that manages the autonomous lifecycle of the SaaS.

Responsibilities:
1. Hourly Loop: Wakes up every hour.
2. Data Fetch: Updates market data using DataEngine.
3. Prediction: Runs Specialist Models via InferenceEngine.
4. Dashboard Update: (Auto-handled via DB update).
5. Alerts: Sends High-Precision (>88%) Telegram signals.
"""

import time
import logging
import schedule
import os
import psutil
from datetime import datetime
from core.inference import InferenceEngine
from core.notifications import NotificationManager
from core.guardrail import get_guardrail

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("executive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Executive")

def job(exec_engine, notifier):
    """The High-Frequency Job."""
    # ── 0. Safety Guardrail (Prop Firm Compliance) ─────────────────
    guard = get_guardrail()
    status = guard.get_safety_status()
    
    if not status['safe']:
        logger.warning(f"🛑 SAFETY HALT: {status['reason']} (Drawdown: {status['drawdown']:.1f}%)")
        return # Skip this cycle

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"⏰ Tick! Starting High-Frequency Cycle (Memory: {mem_mb:.1f} MB)...")
    
    try:
        # ── 1. Precise Signal Resolution (Watchdog) ───────────────────────
        # We now use the ROBUST ExecutiveEngine logic for consistency
        exec_engine.monitor_active_signals()

        # ── 2. Run New Inference Scan (Background) ──────────────────────
        logger.info("🧠 Starting comprehensive 31-pair market scan...")
        # Note: We use ExecutiveEngine's run_scan for full alert/cooldown support
        symbols = exec_engine.get_all_pairs()
        exec_engine.run_scan(symbols)
        
    except Exception as e:
        logger.error(f"❌ Cycle Failed: {e}", exc_info=True)

    logger.info("Cycle Complete. Sleeping...")

def main():
    logger.info("🚀 ApexForex Executive Starting...")
    
    # Initialize the Unified Executive Engine
    # This engine handles both the Inference and the Resolution
    from core.executive import ExecutiveEngine
    exec_engine = ExecutiveEngine(target_win_rate="60%")
    notifier = exec_engine.notifier
    
    # Run once immediately for startup test
    job(exec_engine, notifier)
    
    # Schedule for every 5 minutes (Institutional Standard)
    schedule.every(5).minutes.do(job, exec_engine, notifier)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
