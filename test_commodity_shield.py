import unittest
from core.symbol_guard import is_commodity, is_symbol_blocked

class TestCommodityShield(unittest.TestCase):

    def test_commodity_identification(self):
        """Test commodity detection across various broker formats."""
        commodities = [
            'XAUUSD', 'GOLD', 'xauusd', 'gold', 'XAUUSD.cash', 'XAUUSD.m',
            'XAGUSD', 'SILVER', 'xagusd', 'silver',
            'USOIL', 'USOIL.cash', 'UKOIL', 'UKOIL.cash', 'BRENT', 'WTI', 'CrudeOIL',
            'COPPER', 'XPTUSD', 'XPDUSD', 'NGAS', 'NATGAS'
        ]
        for sym in commodities:
            self.assertTrue(is_commodity(sym), f"Failed to identify commodity: {sym}")
            self.assertTrue(is_symbol_blocked(sym), f"Failed to block commodity: {sym}")

    def test_forex_pairs_allowed(self):
        """Test that regular Forex pairs are NOT blocked."""
        forex_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
            'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
            'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
            'NZDJPY', 'NZDCHF', 'NZDCAD',
            'CADJPY', 'CADCHF', 'CHFJPY', 'USDSGD'
        ]
        for sym in forex_pairs:
            self.assertFalse(is_commodity(sym), f"Incorrectly marked forex as commodity: {sym}")
            self.assertFalse(is_symbol_blocked(sym), f"Incorrectly blocked forex pair: {sym}")

    def test_executive_place_trade_blocks_commodity(self):
        """Test that place_mt5_trade refuses to place commodity orders."""
        from core.executive import ExecutiveEngine
        engine = ExecutiveEngine()
        
        # Test Gold signal with 80% conviction
        gold_signal = {
            'symbol': 'XAUUSD',
            'signal': 'BUY',
            'confidence': 0.80,
            'confidence_tier': 80,
            'price_at_signal': 2500.0,
            'sl_price': 2480.0,
            'tp_price': 2530.0,
            'is_hidden': 0
        }
        res = engine.place_mt5_trade(gold_signal)
        self.assertFalse(res)
        self.assertEqual(gold_signal['is_hidden'], 1)

if __name__ == '__main__':
    unittest.main()
