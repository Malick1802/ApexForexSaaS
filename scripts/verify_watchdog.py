import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.executive import ExecutiveEngine
import logging

logging.basicConfig(level=logging.INFO)

def verify_audusd():
    engine = ExecutiveEngine()
    print("\n--- Starting Watchdog for AUDUSD ---")
    # We only want to check the watchdog, which calls get_active_signals()
    # It will fetch AUDUSD from signals.db and check against TwelveData 1m
    engine.monitor_active_signals()
    print("--- Watchdog Complete ---\n")

if __name__ == "__main__":
    verify_audusd()
