import unittest
from datetime import datetime, timezone
from core.market_hours import is_weekend_halt, get_market_status, get_ny_time

class TestSundayCoolOffBuffer(unittest.TestCase):

    def test_summer_edt_sunday_buffer(self):
        """Test Sunday 2-hour cool-off buffer during Summer (EDT = UTC-4)."""
        # Sunday Aug 23, 2026 17:00 EDT (21:00 UTC) -> Broker Open (HALTED)
        t_1700 = datetime(2026, 8, 23, 21, 0, tzinfo=timezone.utc)
        halted, reason = is_weekend_halt(t_1700)
        self.assertTrue(halted, f"Expected halted at 17:00 EDT but got {halted}")
        self.assertIn("17:00", reason)

        # Sunday Aug 23, 2026 18:30 EDT (22:30 UTC) -> 1.5h in buffer (HALTED)
        t_1830 = datetime(2026, 8, 23, 22, 30, tzinfo=timezone.utc)
        halted, reason = is_weekend_halt(t_1830)
        self.assertTrue(halted, f"Expected halted at 18:30 EDT but got {halted}")

        # Sunday Aug 23, 2026 19:00 EDT (23:00 UTC) -> 2h buffer ends, TRADING ACTIVE
        t_1900 = datetime(2026, 8, 23, 23, 0, tzinfo=timezone.utc)
        halted, reason = is_weekend_halt(t_1900)
        self.assertFalse(halted, f"Expected active at 19:00 EDT but got halted: {reason}")
        self.assertEqual(reason, "OK")

    def test_winter_est_sunday_buffer(self):
        """Test Sunday 2-hour cool-off buffer during Winter (EST = UTC-5)."""
        # Sunday Jan 17, 2027 17:00 EST (22:00 UTC) -> Broker Open (HALTED)
        t_1700 = datetime(2027, 1, 17, 22, 0, tzinfo=timezone.utc)
        halted, reason = is_weekend_halt(t_1700)
        self.assertTrue(halted, f"Expected halted at 17:00 EST but got {halted}")

        # Sunday Jan 17, 2027 18:59 EST (23:59 UTC) -> In buffer (HALTED)
        t_1859 = datetime(2027, 1, 17, 23, 59, tzinfo=timezone.utc)
        halted, reason = is_weekend_halt(t_1859)
        self.assertTrue(halted, f"Expected halted at 18:59 EST but got {halted}")

        # Sunday Jan 17, 2027 19:00 EST (00:00 UTC Jan 18) -> 2h buffer ends, TRADING ACTIVE
        t_1900 = datetime(2027, 1, 18, 0, 0, tzinfo=timezone.utc)
        halted, reason = is_weekend_halt(t_1900)
        self.assertFalse(halted, f"Expected active at 19:00 EST but got halted: {reason}")
        self.assertEqual(reason, "OK")

    def test_friday_and_saturday(self):
        """Test Friday cutoff and Saturday full halt."""
        # Friday 09:59 EDT (13:59 UTC) -> Entry allowed (before 10:00 AM NY cutoff)
        t_fri_ok = datetime(2026, 8, 21, 13, 59, tzinfo=timezone.utc)
        halted, _ = is_weekend_halt(t_fri_ok)
        self.assertFalse(halted)

        # Friday 10:00 EDT (14:00 UTC) -> Halted (10:00 AM NY cutoff reached)
        t_fri_halt = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc)
        halted, _ = is_weekend_halt(t_fri_halt)
        self.assertTrue(halted)

        # Saturday 12:00 UTC -> Halted
        t_sat = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
        halted, _ = is_weekend_halt(t_sat)
        self.assertTrue(halted)

if __name__ == '__main__':
    unittest.main()
