"""
ApexForex Dashboard Auto-Restart Wrapper
Keeps the Streamlit dashboard alive indefinitely, logging crashes.

Usage: python scripts/start_dashboard.py
"""
import subprocess
import time
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging for this wrapper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - DASHBOARD_WATCHDOG - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "dashboard_watchdog.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

DASHBOARD_SCRIPT = str(PROJECT_ROOT / "dashboard" / "app.py")
PORT = 8505
RESTART_DELAY = 5  # seconds between restarts

def run():
    attempts = 0
    while True:
        attempts += 1
        logger.info(f"🚀 Starting Dashboard (Attempt #{attempts})...")

        log_file = LOG_DIR / f"dashboard_startup_{PORT}.log"
        with open(log_file, "a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run",
                    DASHBOARD_SCRIPT,
                    f"--server.port={PORT}",
                    "--server.headless=true",
                    "--logger.level=warning",
                ],
                cwd=str(PROJECT_ROOT),
                stdout=logf,
                stderr=logf,
            )
            logger.info(f"▶ Dashboard PID {proc.pid} running on port {PORT}")

            ret = proc.wait()
            if ret == 0:
                logger.info("✅ Dashboard exited cleanly.")
                break
            else:
                logger.warning(f"⚠️ Dashboard crashed (exit code {ret}). Restarting in {RESTART_DELAY}s...")
                time.sleep(RESTART_DELAY)

if __name__ == "__main__":
    run()
