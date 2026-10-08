import unittest
from unittest.mock import MagicMock, patch
import sys, os
from collections import namedtuple

sys.path.insert(0, r"c:\Users\artem\Downloads\ApexForexSaaS")
from core.executive import ExecutiveEngine

MockPosition = namedtuple('MockPosition', ['symbol', 'ticket', 'type', 'volume'])

class TestCurrencyBasketCap(unittest.TestCase):
    
    def setUp(self):
        with patch('core.executive.InferenceEngine'), \
             patch('core.executive.SignalDatabase'), \
             patch('core.executive.NotificationManager'), \
             patch('core.performance_gate.PerformanceGate'):
            self.engine = ExecutiveEngine()
            
    def test_basket_cap_empty_positions(self):
        """When no positions are open, any valid pair should be allowed."""
        mock_mt5 = MagicMock()
        mock_mt5.positions_get.return_value = ()
        
        with patch.object(ExecutiveEngine, 'mt5', mock_mt5):
            blocked, reason = self.engine._check_currency_basket_cap('GBPJPY')
            self.assertFalse(blocked)
            self.assertEqual(reason, "OK")

    def test_basket_cap_blocks_matching_quote_currency(self):
        """If GBPJPY is open, CADJPY should be blocked (JPY already in use)."""
        mock_mt5 = MagicMock()
        mock_mt5.positions_get.return_value = [
            MockPosition(symbol='GBPJPY', ticket=101, type=0, volume=0.2)
        ]
        
        with patch.object(ExecutiveEngine, 'mt5', mock_mt5):
            blocked, reason = self.engine._check_currency_basket_cap('CADJPY')
            self.assertTrue(blocked)
            self.assertIn("CURRENCY_BASKET_CAP: JPY", reason)

    def test_basket_cap_blocks_matching_base_currency(self):
        """If GBPJPY is open, GBPAUD should be blocked (GBP already in use)."""
        mock_mt5 = MagicMock()
        mock_mt5.positions_get.return_value = [
            MockPosition(symbol='GBPJPY', ticket=101, type=0, volume=0.2)
        ]
        
        with patch.object(ExecutiveEngine, 'mt5', mock_mt5):
            blocked, reason = self.engine._check_currency_basket_cap('GBPAUD')
            self.assertTrue(blocked)
            self.assertIn("CURRENCY_BASKET_CAP: GBP", reason)

    def test_basket_cap_allows_unrelated_currencies(self):
        """If GBPJPY is open, EURUSD should be allowed."""
        mock_mt5 = MagicMock()
        mock_mt5.positions_get.return_value = [
            MockPosition(symbol='GBPJPY', ticket=101, type=0, volume=0.2)
        ]
        
        with patch.object(ExecutiveEngine, 'mt5', mock_mt5):
            blocked, reason = self.engine._check_currency_basket_cap('EURUSD')
            self.assertFalse(blocked)
            self.assertEqual(reason, "OK")

    def test_max_open_trades_cap(self):
        """If 3 positions are open, a 4th position must be blocked even if currencies differ."""
        mock_mt5 = MagicMock()
        mock_mt5.positions_get.return_value = [
            MockPosition(symbol='GBPJPY', ticket=101, type=0, volume=0.2),
            MockPosition(symbol='EURUSD', ticket=102, type=0, volume=0.2),
            MockPosition(symbol='AUDNZD', ticket=103, type=0, volume=0.2),
        ]
        
        with patch.object(ExecutiveEngine, 'mt5', mock_mt5):
            blocked, reason = self.engine._check_currency_basket_cap('CADCHF')
            self.assertTrue(blocked)
            self.assertIn("MAX_OPEN_REACHED", reason)

    def test_dynamic_risk_scaling(self):
        """Verify dynamic lot sizing scales off account balance."""
        mock_mt5 = MagicMock()
        mock_account = MagicMock()
        mock_account.balance = 9157.25
        mock_mt5.account_info.return_value = mock_account
        
        mock_sym_info = MagicMock()
        mock_sym_info.ask = 1.08500
        mock_sym_info.volume_step = 0.01
        mock_sym_info.volume_min = 0.01
        mock_sym_info.volume_max = 50.0
        mock_mt5.symbol_info.return_value = mock_sym_info
        
        # Native calc profit: 25 pips on EURUSD = $250 for 1.0 standard lot
        mock_mt5.order_calc_profit.return_value = -250.0
        mock_mt5.order_calc_margin.return_value = 3300.0
        self.engine.inference_engine.data_engine.get_pip_value.return_value = 0.0001
        
        with patch.object(ExecutiveEngine, 'mt5', mock_mt5):
            lots = self.engine._calculate_lot_size('EURUSD', 25.0)
            # Risk = 9157.25 * 0.005 = $45.786
            # Lots = 45.786 / 250 = 0.183 -> rounded to step 0.01 = 0.18 lots
            self.assertEqual(lots, 0.18)

if __name__ == '__main__':
    unittest.main()
