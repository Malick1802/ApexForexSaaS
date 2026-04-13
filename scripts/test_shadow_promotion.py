import os
import sys
import logging
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Set up paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.core.inference import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ShadowTest')

def test_shadow_promotion():
    logger.info("Starting Shadow Promotion Verification Test...")
    
    # Create engine instance
    # We mock out the __init__ to avoid any real initialization
    with patch.object(InferenceEngine, '__init__', return_value=None):
        engine = InferenceEngine()
        # Manually set up only what's needed for predict_symbol
        engine.perf_gate = MagicMock()
        engine.data_engine = MagicMock()
        engine.feature_engineer = MagicMock()
        engine.global_engineer = MagicMock()
        engine._regime_detector = MagicMock()
        engine.calibrator = MagicMock()
        engine.db = MagicMock()
        engine._model_cache = {}
        
        # Mocking the flow
        engine._is_data_stale = MagicMock(return_value=False)
        engine._update_global_context = MagicMock(return_value={})
        engine._regime_detector.is_tradeable.return_value = (True, 0.70, MagicMock())
        engine.calculate_tp_sl = MagicMock(return_value={'tp_price': 1.1, 'sl_price': 0.9, 'tp_pips': 100, 'sl_pips': 100})
        engine.calculate_lots_precision = MagicMock(return_value=0.1)
        
        df = pd.DataFrame({
            'close': [1.0]*100, 'low': [0.99]*100, 'high': [1.01]*100, 'atr_norm': [0.01]*100
        }, index=pd.date_range('2026-04-07', periods=100, freq='h', tz='UTC'))
        
        engine.data_engine.fetch.return_value = df
        engine.feature_engineer.extract_features.return_value = df
        engine.global_engineer.add_global_features.return_value = df
        engine.feature_engineer.create_sequences.return_value = (np.random.rand(10, 15, 63), None)
        engine.calibrator.calibrate.side_effect = lambda s, sig, conf: conf
        
        # 1. Setup Benched State in mock gate
        engine.perf_gate.get_tier_status.return_value = 'BENCHED'
        engine.perf_gate.is_tier_approved.return_value = False
        
        # 2. Setup "Stalled" Model (High threshold vs mediocre prob)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.3, 0.62, 0.08]]) # 62% BUY
        mock_model.input_shape = (None, 15, 63)
        
        # This is where we simulate the "WAIT" logic
        engine.load_phase3_expert.return_value = None
        engine.load_foundation_model.return_value = None
        engine.load_models.return_value = {
            'model': mock_model, 
            'scaler': MagicMock(n_features_in_=63), 
            'buy_threshold': 0.85, 
            'sell_threshold': 0.85, 
            'model_type': 'enhanced',
            'model_trades': 50
        }
        
        # 3. Run Prediction
        logger.info("Running prediction for AUDNZD with 62% BUY setup (Benched)...")
        # In real logic: 62% < 85% threshold -> initially WAIT.
        # But is_tier_benched = True and 62% >= 60% -> is_authorized = True.
        # My new fix should promote it back to BUY.
        
        result = engine.predict_symbol('AUDNZD', save_to_db=False, win_rate='70%')
        
        if result:
            logger.info(f"Final Signal: {result['signal']}")
            logger.info(f"Final Confidence: {result['confidence']:.1%}")
            logger.info(f"Is Hidden (Shadow): {result['is_hidden']}")
            
            if result['signal'] == 'BUY' and result['is_hidden'] == 1:
                logger.info("✅ SUCCESS: Wait signal correctly promoted to BUY for shadow trade!")
            else:
                logger.error(f"❌ FAIL: Expected shadow BUY but got {result['signal']} (Hidden={result['is_hidden']})")
        else:
            logger.error("❌ FAIL: Prediction returned None.")

if __name__ == '__main__':
    test_shadow_promotion()
