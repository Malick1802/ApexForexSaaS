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

import sys
import io

# Force UTF-8 for console output on Windows to prevent Emoji crashes
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
        logging.FileHandler("executive.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Executive")

def job(exec_engine, notifier):
    """The High-Frequency Job."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"⏰ Tick! Starting High-Frequency Cycle (Memory: {mem_mb:.1f} MB)...")
    
    # ── 1. Precise Signal Resolution (Watchdog) ───────────────────────
    # Run active signal monitoring first and unconditionally (so we resolve exits even during weekend/drawdown halts)
    try:
        logger.info("Watchdog: Syncing active signals with market reality...")
        exec_engine.monitor_active_signals()
    except Exception as e:
        logger.error(f"❌ Watchdog Sync Failed: {e}", exc_info=True)

    # ── 1b. Dynamic Model Whitelist Daily Rollover Check ─────────────
    try:
        from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
        dw_mgr = get_dynamic_whitelist_manager()
        if dw_mgr.check_daily_refresh():
            logger.info("🔄 Autonomous Daily Rollover: Dynamic YTD winning assets updated for all active models.")
    except Exception as _dwe:
        logger.error(f"❌ Dynamic Whitelist daily rollover check failed: {_dwe}", exc_info=True)

    # ── 2. Safety Guardrail (Prop Firm Compliance & Emergency Kill Switch) ─
    try:
        guard = get_guardrail()
        status = guard.get_safety_status(exec_engine=exec_engine)
        if not status['safe']:
            logger.warning(f"🛑 SAFETY HALT: {status['reason']} (Drawdown: {status['drawdown']:.1f}%) - Skipping Inference Scan.")
            return # Skip new trade entries
    except Exception as ge:
        logger.error(f"❌ Safety Guardrail Check Failed: {ge}", exc_info=True)
        return

    from core.market_hours import is_weekend_halt, is_crypto
    halted, reason = is_weekend_halt()
    symbols = exec_engine.get_all_pairs()
    if halted:
        symbols = [s for s in symbols if is_crypto(s)]
        if not symbols:
            logger.info(f"⏸️ WEEKEND MODE: {reason}. No active crypto pairs to scan.")
            return
        logger.info(f"🪙 WEEKEND 24/7 CRYPTO SCAN: Scanning {len(symbols)} crypto pairs during weekend ({symbols}).")

    try:
        # ── 3. Run New Inference Scan (Background) ──────────────────────
        logger.info(f"🧠 Starting market scan across {len(symbols)} pairs...")
        # Note: We use ExecutiveEngine's run_scan for full alert/cooldown support
        exec_engine.run_scan(symbols)
        
    except Exception as e:
        logger.error(f"❌ Inference Scan Cycle Failed: {e}", exc_info=True)

    logger.info("Cycle Complete. Sleeping...")

def main():
    logger.info("🚀 ApexForex Executive Starting...")

    # Singleton Process Mutex Check: Prevent duplicate background daemons from running
    my_pid = os.getpid()
    my_ppid = os.getppid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] not in (my_pid, my_ppid) and 'python' in (proc.info['name'] or '').lower():
                cmd = " ".join(proc.info['cmdline'] or []).lower()
                if "main.py" in cmd:
                    logger.critical(f"🛑 DUPLICATE DAEMON BLOCKED: Another instance of main.py is already running (PID {proc.info['pid']}). Exiting to prevent duplicate orders.")
                    print(f"\n[CRITICAL] Another instance of main.py is already running (PID {proc.info['pid']})! Exiting to prevent duplicate orders.")
                    sys.exit(0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "executive.pid")
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
    
    # Prevent Windows from entering idle sleep while executive is active
    try:
        import ctypes
        # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        logger.info("⚡ Windows Sleep Prevention Active: System will not sleep while executive is running.")
    except Exception as _se:
        logger.debug(f"Sleep prevention init: {_se}")

    # Disable Windows Console QuickEdit mode to prevent clicking from freezing execution
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hStdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):
            new_mode = (mode.value & ~0x0040) | 0x0080  # Clear ENABLE_QUICK_EDIT_MODE, set ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(hStdin, new_mode)
            logger.info("⚡ Windows Console QuickEdit disabled: Clicking inside window will not pause execution.")
    except Exception as _qe:
        logger.debug(f"QuickEdit disable init: {_qe}")

    # Initialize the Unified Executive Engine
    # This engine handles both the Inference and the Resolution
    from core.executive import ExecutiveEngine
    exec_engine = ExecutiveEngine(target_win_rate="61%")
    notifier = exec_engine.notifier
    
    # Start the M15 Armed Sniper background watcher immediately
    try:
        from core.manual_model import start_armed_sniper_watcher
        start_armed_sniper_watcher()
        logger.info("🎯 M15 Armed Sniper Watcher initialized in Executive.")
    except Exception as _we:
        logger.warning(f"Could not start M15 sniper watcher in executive: {_we}")

    # Start the Automated Confluence Day/Swing M15 Watcher immediately
    try:
        from core.confluence_model import start_confluence_scanner_watcher
        start_confluence_scanner_watcher()
        logger.info("⚡ Confluence Day/Swing Automated Watcher initialized in Executive.")
    except Exception as _ce:
        logger.warning(f"Could not start Confluence watcher in executive: {_ce}")

    # Initialize & Synchronize Dynamic Model Whitelists YTD on startup
    try:
        from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
        dw_mgr = get_dynamic_whitelist_manager()
        dw_mgr.load_from_disk()
        logger.info("🌟 Dynamic Model Whitelist Engine synchronized on startup.")
    except Exception as _dwe:
        logger.error(f"Startup dynamic whitelist sync failed: {_dwe}")

    # Run once immediately for startup test
    job(exec_engine, notifier)
    
    # Schedule for every 1 minute (Real-Time Precision Execution)
    schedule.every(1).minutes.do(job, exec_engine, notifier)
    
    # Schedule Autonomous 00:01 UTC Daily Dynamic Whitelist Recalculation
    try:
        from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
        schedule.every().day.at("00:01").do(get_dynamic_whitelist_manager().compute_ytd_whitelists)
        logger.info("🔄 Autonomous Daily Dynamic Whitelist Recalculation scheduled for 00:01 UTC daily.")
    except Exception as _se:
        logger.warning(f"Could not schedule dynamic whitelist daily recalculation: {_se}")

    # Schedule Daily & Weekly Automated Telegram Performance Scorecards
    # 22:00 UTC (NY Close / Asian Open) daily summary
    schedule.every().day.at("22:00").do(notifier.send_periodic_performance_report, "both", 50.0)
    # Sunday 21:00 UTC weekly kick-off scorecard
    schedule.every().sunday.at("21:00").do(notifier.send_periodic_performance_report, "weekly", 50.0)

    logger.info("📅 Performance Scorecards scheduled for 22:00 UTC daily and Sunday 21:00 UTC.")
    
    while True:
        try:
            schedule.run_pending()
        except Exception as loop_err:
            logger.critical(f"🚨 Unhandled exception in scheduler loop: {loop_err}", exc_info=True)
            time.sleep(5)
        time.sleep(1)

if __name__ == "__main__":
    main()
