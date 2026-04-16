import logging
import sqlite3
import pandas as pd
from core.executive import ExecutiveEngine
from core.performance_gate import PerformanceGate

# Setup logging
logging.basicConfig(level=logging.INFO)

def fix_and_report():
    print("\n--- TRIGGERING WATCHDOG (FIXED EDITION) ---")
    engine = ExecutiveEngine()
    
    # This will now include hidden signals thanks to the fix
    engine.monitor_active_signals()
    
    print("\n--- RESOLUTION STATUS FOR CRUDEOIL ---")
    conn = sqlite3.connect('signals.db')
    query = """
    SELECT id, symbol, signal, outcome, timestamp 
    FROM signals 
    WHERE symbol = 'CrudeOIL' 
    ORDER BY timestamp DESC 
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    
    print("\n--- RECOMPUTING PERFORMANCE MATRIX ---")
    gate = PerformanceGate()
    gate.recompute_from_db(lookback_days=14)
    gate.save_whitelist()
    
    # Accessing the correct attribute: performance_matrix
    if 'CrudeOIL' in gate.performance_matrix:
        stats = gate.performance_matrix['CrudeOIL']
        print(f"NEW CrudeOIL PERFORMANCE STATS:")
        # Pretty print the nest BUY/SELL details
        for direction, tiers in stats.items():
            for tier, data in tiers.items():
                print(f"  {direction} @ {tier}% Tier: {data['accuracy']:.1%} accuracy ({data['trades']} trades)")
    else:
        print("CrudeOIL still has no resolved trades for the performance matrix.")
        
    conn.close()

if __name__ == "__main__":
    fix_and_report()
