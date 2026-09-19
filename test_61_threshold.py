import unittest
import sys
import os
from datetime import datetime, timezone
sys.path.insert(0, os.path.abspath("."))

from core.symbol_guard import is_symbol_blocked, is_commodity, is_direction_blocked
from core.market_hours import is_friday_trade_entry_allowed, is_weekend_halt
from core.guardrail import PropGuardrail

class TestTradingRulesAndSafeguards(unittest.TestCase):
    def test_commodity_guard(self):
        # All commodities are permanently banned
        self.assertTrue(is_symbol_blocked("XAUUSD"))
        self.assertTrue(is_symbol_blocked("USOIL.cash"))
        # Exotics still blocked in config
        self.assertTrue(is_symbol_blocked("COPPER"))
        self.assertTrue(is_symbol_blocked("XAGUSD"))
        # Forex unblocked
        self.assertFalse(is_symbol_blocked("EURUSD"))
        self.assertFalse(is_symbol_blocked("GBPJPY"))
        # Commodities recognized by helper
        self.assertTrue(is_commodity("XAUUSD"))
        self.assertTrue(is_commodity("USOIL.cash"))
        self.assertFalse(is_commodity("EURUSD"))

    def test_directional_guard(self):
        # Directional blacklists preserved
        self.assertTrue(is_direction_blocked("EURUSD", "BUY"))
        self.assertFalse(is_direction_blocked("EURUSD", "SELL"))
        self.assertTrue(is_direction_blocked("EURCAD", "BUY"))
        self.assertFalse(is_direction_blocked("EURCAD", "SELL"))
        self.assertTrue(is_direction_blocked("AUDUSD", "BUY"))
        self.assertFalse(is_direction_blocked("AUDUSD", "SELL"))

    def test_friday_14_utc_cutoff(self):
        # Friday before 14:00 UTC (e.g. 13:59 UTC) -> Allowed
        fri_early = datetime(2026, 9, 11, 13, 59, tzinfo=timezone.utc)
        allowed, reason = is_friday_trade_entry_allowed(fri_early)
        self.assertTrue(allowed)

        # Friday at or after 14:00 UTC (e.g. 14:00 UTC) -> Blocked
        fri_late = datetime(2026, 9, 11, 14, 0, tzinfo=timezone.utc)
        allowed, reason = is_friday_trade_entry_allowed(fri_late)
        self.assertFalse(allowed)
        self.assertIn("14:00 UTC", reason)

        # Non-Friday (e.g. Thursday 15:00 UTC) -> Allowed
        thu = datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)
        allowed, reason = is_friday_trade_entry_allowed(thu)
        self.assertTrue(allowed)

    def test_config_settings(self):
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get('notifications', {}).get('telegram', {}).get('alert_threshold'), 0.61)
        self.assertEqual(cfg.get('trading', {}).get('target_win_rate'), "61%")
        # max_open_trades is now 0 (uncapped)
        self.assertEqual(cfg.get('mt5', {}).get('max_open_trades'), 0)
        # Safety settings
        safety = cfg.get('safety', {})
        self.assertEqual(safety.get('max_daily_drawdown_amount'), 450.0)
        self.assertEqual(safety.get('friday_entry_cutoff_utc_hour'), 14)

if __name__ == '__main__':
    unittest.main()
