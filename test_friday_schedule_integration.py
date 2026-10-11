import unittest
from datetime import datetime, timezone
import zoneinfo

from core.market_hours import (
    get_ny_time,
    is_friday_trade_entry_allowed,
    is_friday_auto_exit_time,
    is_friday_auto_exit_window,
    is_metal,
    is_crypto,
    is_weekend_halt,
    get_market_status
)
from core.guardrail import PropGuardrail

class TestFridayScheduleAndSeasons(unittest.TestCase):

    def test_summer_edt_schedule(self):
        """Test Summer (EDT, UTC-4) schedule where 17:00 EDT = 21:00 UTC, 16:00 EDT = 20:00 UTC."""
        # Friday 09:55 EDT (13:55 UTC) -> Still allowed (before 10:00 AM NY cutoff)
        t_allow = datetime(2026, 7, 17, 13, 55, tzinfo=timezone.utc)
        allowed, _ = is_friday_trade_entry_allowed(t_allow)
        self.assertTrue(allowed)
        self.assertFalse(is_friday_auto_exit_time(t_allow))
        halt, _ = is_weekend_halt(t_allow)
        self.assertFalse(halt)

        # Friday 10:05 EDT (14:05 UTC) -> Cutoff reached (10:00 AM NY cutoff)
        t_cutoff = datetime(2026, 7, 17, 14, 5, tzinfo=timezone.utc)
        allowed, reason = is_friday_trade_entry_allowed(t_cutoff)
        self.assertFalse(allowed)
        self.assertIn("FRIDAY_CUTOFF", reason)
        self.assertFalse(is_friday_auto_exit_time(t_cutoff))
        halt, _ = is_weekend_halt(t_cutoff)
        self.assertTrue(halt)

        # Friday 15:54 EDT (19:54 UTC) -> Just before Metals 5-min exit cutoff (15:55 EDT)
        t_pre_metals = datetime(2026, 7, 17, 19, 54, tzinfo=timezone.utc)
        self.assertFalse(is_friday_auto_exit_time(t_pre_metals, symbol="XAUUSD"))
        self.assertFalse(is_friday_auto_exit_window(t_pre_metals))

        # Friday 15:56 EDT (19:56 UTC) -> Metals Exit Triggered (5 min before 16:00 EDT close)
        t_metals_exit = datetime(2026, 7, 17, 19, 56, tzinfo=timezone.utc)
        self.assertTrue(is_friday_auto_exit_time(t_metals_exit, symbol="XAUUSD"))
        self.assertTrue(is_friday_auto_exit_time(t_metals_exit, symbol="XAGUSD"))
        self.assertTrue(is_friday_auto_exit_window(t_metals_exit))
        # Forex has NOT reached exit at 15:56 (closes at 17:00, exit at 16:55)
        self.assertFalse(is_friday_auto_exit_time(t_metals_exit, symbol="EURUSD"))
        self.assertFalse(is_friday_auto_exit_time(t_metals_exit))

        # Friday 16:35 EDT (20:35 UTC) -> Old 30-min exit time; now Forex keeps trading until 16:55!
        t_mid = datetime(2026, 7, 17, 20, 35, tzinfo=timezone.utc)
        self.assertFalse(is_friday_auto_exit_time(t_mid, symbol="EURUSD"))
        self.assertTrue(is_friday_auto_exit_time(t_mid, symbol="XAUUSD"))

        # Friday 16:56 EDT (20:56 UTC) -> Forex Auto-Exit Triggered (5 min before 17:00 EDT close)
        t_exit = datetime(2026, 7, 17, 20, 56, tzinfo=timezone.utc)
        self.assertTrue(is_friday_auto_exit_time(t_exit, symbol="EURUSD"))
        self.assertTrue(is_friday_auto_exit_time(t_exit))

        # Crypto remains exempt from Friday auto-exit
        self.assertFalse(is_friday_auto_exit_time(t_exit, symbol="BTCUSD"))

        # Saturday (All day halt for forex, not crypto)
        t_sat = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
        halt, _ = is_weekend_halt(t_sat)
        self.assertTrue(halt)
        halt_crypto, _ = is_weekend_halt(t_sat, symbol="BTCUSD")
        self.assertFalse(halt_crypto)

        # Sunday 16:55 EDT (20:55 UTC) -> Still halted before 17:00 open
        t_sun_before = datetime(2026, 7, 19, 20, 55, tzinfo=timezone.utc)
        halt, _ = is_weekend_halt(t_sun_before)
        self.assertTrue(halt)

        # Sunday 18:55 EDT (22:55 UTC) -> Still halted in 2h cool-off buffer
        t_sun_buffer = datetime(2026, 7, 19, 22, 55, tzinfo=timezone.utc)
        halt, _ = is_weekend_halt(t_sun_buffer)
        self.assertTrue(halt)

        # Sunday 19:05 EDT (23:05 UTC) -> Trading Active (after 2h buffer)!
        t_sun_open = datetime(2026, 7, 19, 23, 5, tzinfo=timezone.utc)
        halt, _ = is_weekend_halt(t_sun_open)
        self.assertFalse(halt)

    def test_winter_est_schedule(self):
        """Test Winter (EST, UTC-5) schedule where 17:00 EST = 22:00 UTC, 16:00 EST = 21:00 UTC."""
        # Friday 09:55 EST (14:55 UTC) -> Still allowed (before 10:00 AM NY cutoff)
        t_allow = datetime(2026, 1, 16, 14, 55, tzinfo=timezone.utc)
        allowed, _ = is_friday_trade_entry_allowed(t_allow)
        self.assertTrue(allowed)
        self.assertFalse(is_friday_auto_exit_time(t_allow))
        halt, _ = is_weekend_halt(t_allow)
        self.assertFalse(halt)

        # Friday 10:05 EST (15:05 UTC) -> Cutoff reached (10:00 AM NY cutoff)
        t_cutoff = datetime(2026, 1, 16, 15, 5, tzinfo=timezone.utc)
        allowed, reason = is_friday_trade_entry_allowed(t_cutoff)
        self.assertFalse(allowed)
        self.assertIn("FRIDAY_CUTOFF", reason)
        self.assertFalse(is_friday_auto_exit_time(t_cutoff))
        halt, _ = is_weekend_halt(t_cutoff)
        self.assertTrue(halt)

        # Friday 15:56 EST (20:56 UTC) -> Metals Exit Triggered (5 min before 16:00 EST close)
        t_metals_exit = datetime(2026, 1, 16, 20, 56, tzinfo=timezone.utc)
        self.assertTrue(is_friday_auto_exit_time(t_metals_exit, symbol="XAUUSD"))
        self.assertFalse(is_friday_auto_exit_time(t_metals_exit, symbol="EURUSD"))

        # Friday 16:56 EST (21:56 UTC) -> Forex Auto-Exit Triggered (5 min before 17:00 EST close)
        t_exit = datetime(2026, 1, 16, 21, 56, tzinfo=timezone.utc)
        self.assertTrue(is_friday_auto_exit_time(t_exit, symbol="EURUSD"))
        self.assertTrue(is_friday_auto_exit_time(t_exit))

        # Sunday 18:55 EST (23:55 UTC) -> Still halted in 2h cool-off buffer
        t_sun_buffer = datetime(2026, 1, 18, 23, 55, tzinfo=timezone.utc)
        halt, _ = is_weekend_halt(t_sun_buffer)
        self.assertTrue(halt)

        # Sunday 19:05 EST (00:05 UTC Jan 19) -> Trading Active (after 2h buffer)!
        t_sun_open = datetime(2026, 1, 19, 0, 5, tzinfo=timezone.utc)
        halt, _ = is_weekend_halt(t_sun_open)
        self.assertFalse(halt)

    def test_metals_and_crypto_helpers(self):
        """Test is_metal and is_crypto helper classifications."""
        self.assertTrue(is_metal("XAUUSD"))
        self.assertTrue(is_metal("XAGUSD"))
        self.assertTrue(is_metal("GOLD"))
        self.assertTrue(is_metal("USOIL.cash"))
        self.assertFalse(is_metal("EURUSD"))
        self.assertFalse(is_metal("GBPUSD"))
        self.assertFalse(is_metal("BTCUSD"))

        self.assertTrue(is_crypto("BTCUSD"))
        self.assertTrue(is_crypto("ETHUSD"))
        self.assertTrue(is_crypto("SOLUSD"))
        self.assertFalse(is_crypto("EURUSD"))
        self.assertFalse(is_crypto("XAUUSD"))

    def test_guardrail_integration(self):
        """Test PropGuardrail integrates with market hours."""
        guard = PropGuardrail()
        status = guard.get_safety_status()
        self.assertIn('safe', status)
        self.assertIn('reason', status)

if __name__ == '__main__':
    unittest.main()
