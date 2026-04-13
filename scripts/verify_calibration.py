import sqlite3
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup
db_path = PROJECT_ROOT / "signals.temp.db"
if db_path.exists(): os.remove(db_path)

def setup_mock_db():
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            signal TEXT,
            confidence REAL,
            outcome TEXT,
            timestamp TEXT
        )
    """)
    
    # 🚨 Scenario: AI is overconfident on EURUSD BUYs
    # It says 90% confidence, but it's only right 50% of the time.
    for _ in range(50):
        # 50 SUCCESS, 50 FAIL
        conn.execute("INSERT INTO signals (symbol, signal, confidence, outcome) VALUES (?, ?, ?, ?)",
                     ("EURUSD", "BUY", 0.90 + np.random.uniform(-0.02, 0.02), "SUCCESS"))
        conn.execute("INSERT INTO signals (symbol, signal, confidence, outcome) VALUES (?, ?, ?, ?)",
                     ("EURUSD", "BUY", 0.90 + np.random.uniform(-0.02, 0.02), "FAIL"))
    
    conn.commit()
    conn.close()

def test_calibration():
    from core.core.calibration import get_calibration_manager
    from core.core.inference import InferenceEngine
    
    print("\n--- Calibration Verification ---")
    
    manager = get_calibration_manager()
    # Train from mock DB
    manager.train_from_database(db_path=str(db_path), min_samples=30)
    
    # Instance InferenceEngine
    engine = InferenceEngine()
    engine.calibrator = manager # Inject mock manager
    
    # Test 1: Raw AI says 90%
    raw_conf = 0.90
    calibrated = engine.calibrator.calibrate("EURUSD", "BUY", raw_conf)
    
    print(f"📊 EURUSD BUY Signal:")
    print(f"  Raw AI Confidence: {raw_conf:.1%}")
    print(f"  Calibrated Accuracy: {calibrated:.1%}")
    
    if calibrated < 0.65:
        print("✅ SUCCESS: Calibration correctly identified the overconfidence!")
    else:
        print("❌ FAILURE: Calibration did not adjust high enough.")

if __name__ == "__main__":
    setup_mock_db()
    test_calibration()
    if db_path.exists(): os.remove(db_path)
