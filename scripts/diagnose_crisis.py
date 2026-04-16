import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup paths
sys.path.append(os.getcwd())

from core.gmm_regime_detector import GMMRegimeDetector
from data_pipeline import DataEngine
from core.database import SignalDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DIAGNOSE")

def run_diag():
    print("=== APEX FOREX CRISIS DIAGNOSTIC ===")
    print(f"Time: {datetime.now()}")
    
    engine = DataEngine()
    detector = GMMRegimeDetector()
    db = SignalDatabase()
    
    symbols = ["EURUSD", "GBPUSD", "AUDUSD"]
    
    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        try:
            # 1. Check current logic result
            df = engine.fetch(symbol, interval='1h', days=30)
            if df is None or len(df) == 0:
                print(f"FAILED to fetch data for {symbol}")
                continue
                
            res = detector.detect(df, symbol)
            print(f"CURRENT CALCULATION:")
            print(f"  Regime: {res.regime}")
            print(f"  Blocked: {res.block_trading}")
            print(f"  Reason: {res.reason}")
            
            # 2. Check Database content
            signals = db.get_recent_signals(symbol=symbol, limit=3)
            print(f"DATABASE RECENT SIGNALS:")
            if not signals:
                print("  No signals found.")
            for s in signals:
                ts = s.get('timestamp', 'N/A')
                reg = s.get('regime', 'N/A')
                direction = s.get('direction', 'N/A')
                print(f"  [{ts}] {direction} (Regime: {reg})")
                
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n=== SYSTEM STATE ===")
    # Check for running python processes (approximate check using OS)
    if os.name == 'nt':
         import subprocess
         try:
             output = subprocess.check_output('tasklist /fi "imagename eq python.exe"', shell=True).decode()
             print(output)
         except:
             print("Could not list processes via tasklist")

if __name__ == "__main__":
    run_diag()
