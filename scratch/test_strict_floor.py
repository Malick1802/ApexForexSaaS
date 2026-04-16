import logging
import unittest
from unittest.mock import MagicMock, patch
from core.inference import InferenceEngine
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestInferenceFloor(unittest.TestCase):
    @patch('core.inference.DataEngine')
    @patch('core.inference.SignalDatabase')
    @patch('core.inference.get_performance_gate')
    @patch('core.inference.get_calibration_manager')
    @patch('core.inference.get_detector')
    @patch('core.inference.NotificationManager')
    @patch('tensorflow.keras.models.load_model')
    def test_calibrated_floor_block(self, mock_load, mock_notif, mock_detector, mock_calib_factory, mock_gate_factory, mock_db, mock_data):
        # Setup mocks
        mock_calib = MagicMock()
        mock_calib_factory.return_value = mock_calib
        
        mock_gate = MagicMock()
        mock_gate_factory.return_value = mock_gate

        engine = InferenceEngine()
        
        # Mock Data
        mock_df = pd.DataFrame({
            'close': [100.0] * 100,
            'atr': [0.1] * 100,
            'atr_norm': [0.1] * 100
        })
        mock_features = pd.DataFrame({'vix_proxy': [1.0], 'yield_curve_slope': [1.0], 'atr_norm': [0.1]})
        mock_data.return_value.get_full_features.return_value = (mock_df, mock_features)
        
        # Mock Model (Expert logic)
        mock_model = MagicMock()
        # Binary model behavior
        mock_model.predict.return_value = np.array([[0.3, 0.7]]) # Raw Buy = 70%
        engine.loaded_models = {
            'EURUSD': {
                'model': mock_model,
                'model_type': 'binary',
                'buy_threshold': 0.60,
                'sell_threshold': 0.60,
                'classes': ['SELL', 'BUY']
            }
        }
        
        # --- TEST 1: RAW 70% -> CALIBRATED 59% (Should BLOCK) ---
        mock_calib.calibrate.return_value = 0.59
        mock_gate.get_tier_status.return_value = "BENCHED"
        
        # Ensure we pass the tradeable check
        engine.data_engine.get_market_regime.return_value = ("NORMAL", 0.0)
        
        result = engine.predict_symbol("EURUSD", win_rate="70%", save_to_db=False, allow_stale=True)
        
        print(f"\n[Test 1] Raw: 70%, Calibrated: 59%")
        if result is None:
            print("Result is None (Unexpected Block)")
        else:
            print(f"Result Signal: {result.get('signal')}")
            print(f"Result Authorized: {result.get('outcome') != 'N/A'}")
            self.assertEqual(result['signal'], "WAIT")
            self.assertEqual(result['outcome'], "N/A")

        # --- TEST 2: RAW 70% -> CALIBRATED 61% (Should AUTHORIZE) ---
        mock_calib.calibrate.return_value = 0.61
        
        result = engine.predict_symbol("EURUSD", win_rate="70%", save_to_db=False, allow_stale=True)
        
        print(f"\n[Test 2] Raw: 70%, Calibrated: 61%")
        if result is None:
            print("Result is None (Unexpected Block)")
        else:
            print(f"Result Signal: {result.get('signal')}")
            print(f"Result Authorized: {result.get('outcome') != 'N/A'}")
            self.assertEqual(result['signal'], "BUY")
            self.assertEqual(result['outcome'], "ACTIVE")

if __name__ == "__main__":
    unittest.main()
