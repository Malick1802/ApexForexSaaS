import logging
from core.executive import ExecutiveEngine

# Explicitly setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

def resolve_all():
    print("\n--- TRIGGERING GLOBAL SIGNAL RESOLUTION SWEEP ---")
    print("This will check all 2-day-old 'Zombies' (like NZDJPY) against 14 days of history.\n")
    
    engine = ExecutiveEngine()
    
    # 1. Manually trigger the resolution loop
    # This now uses the new 14-day lookback logic
    engine.monitor_active_signals()
    
    print("\n--- SWEEP COMPLETE ---\n")

if __name__ == "__main__":
    resolve_all()
