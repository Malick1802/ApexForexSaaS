import sqlite3
import pandas as pd
from core.performance_gate import PerformanceGate

def audit_crude():
    conn = sqlite3.connect('signals.db')
    
    print("--- RECENT CRUDEOIL SIGNALS ---")
    query = """
    SELECT id, symbol, signal, confidence, regime, outcome, timestamp 
    FROM signals 
    WHERE symbol = 'CrudeOIL' 
    ORDER BY timestamp DESC 
    LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    
    print("\n--- CURRENT PERFORMANCE MATRIX (WIN RATES) ---")
    gate = PerformanceGate()
    matrix = gate.matrix
    
    if 'CrudeOIL' in matrix:
        stats = matrix['CrudeOIL']
        print(f"Stats for CrudeOIL: {stats}")
    else:
        print("CrudeOIL not yet in performance matrix (insufficient resolved trades).")
        
    conn.close()

if __name__ == "__main__":
    audit_crude()
