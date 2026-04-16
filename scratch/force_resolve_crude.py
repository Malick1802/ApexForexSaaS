import logging
from core.executive import ExecutiveEngine

# Explicitly setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

def resolve_and_stats():
    engine = ExecutiveEngine()
    
    print("\n--- TRIGGERING ACTIVE SIGNAL RESOLUTION CHECK ---")
    engine.monitor_active_signals()
    
    print("\n--- UPDATED RECENT SIGNALS FOR CRUDEOIL ---")
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect('signals.db')
    df = pd.read_sql_query("SELECT id, symbol, signal, outcome, timestamp FROM signals WHERE symbol = 'CrudeOIL' ORDER BY timestamp DESC LIMIT 5", conn)
    print(df.to_string(index=False))
    
    print("\n--- UPDATING PERFORMANCE MATRIX ---")
    from core.performance_gate import PerformanceGate
    gate = PerformanceGate()
    gate.recompute_from_db(lookback_days=14)
    gate.save_whitelist()
    
    if 'CrudeOIL' in gate.matrix:
        print(f"NEW CrudeOIL STATS: {gate.matrix['CrudeOIL']}")
    else:
        print("CrudeOIL still has no resolved trades in the last 14 days.")
    
    conn.close()

if __name__ == "__main__":
    resolve_and_stats()
