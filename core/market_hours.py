"""
core/market_hours.py
====================
Centralized Forex Market Hours & Seasonal Timezone Management.

Forex Market Standard:
- Friday Market Close:  17:00 (5:00 PM) New York time (US Eastern).
- Sunday Market Open:   17:00 (5:00 PM) New York time (US Eastern).
- Friday Entry Cutoff:  10:00 (10:00 AM) New York time.
- Friday Auto-Exit:     16:30 (4:30 PM) New York time (30 minutes before close).

Timezone Handling:
Uses Python's zoneinfo (`America/New_York`) to automatically adapt to seasonal
Daylight Saving Time transitions (EDT UTC-4 in Summer vs EST UTC-5 in Winter).
"""

import zoneinfo
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def get_ny_time(dt_utc: datetime = None) -> datetime:
    """Convert UTC datetime to New York time with full DST support."""
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    elif dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(NY_TZ)

def is_friday_trade_entry_allowed(dt_utc: datetime = None) -> Tuple[bool, str]:
    """
    Check if new trade / signal generation is allowed on Friday.
    Stops generating new signals on Friday at 10:00 AM NY time (14:00 UTC EDT / 15:00 UTC EST).
    """
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    elif dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    ny = get_ny_time(dt_utc)
    if ny.weekday() == 4 and ny.hour >= 10:
        return False, f"FRIDAY_CUTOFF: After 14:00 UTC / 10:00 AM NY ({ny.strftime('%H:%M %Z')}, Friday cutoff reached)"
    if ny.weekday() in (5, 6):
        return False, "WEEKEND_HALT: Market closed for the weekend"
    
    return True, "OK"

def is_friday_auto_exit_time(dt_utc: datetime = None) -> bool:
    """
    Check if Friday Auto-Exit should trigger.
    Triggers 30 minutes before market close (Friday 16:30 New York time).
    """
    ny = get_ny_time(dt_utc)
    if ny.weekday() == 4:
        if ny.hour > 16 or (ny.hour == 16 and ny.minute >= 30):
            return True
    return False

def is_weekend_halt(dt_utc: datetime = None) -> Tuple[bool, str]:
    """
    Check if system is in weekend halt mode:
    - Friday: From 10:00 AM NY time (Friday cutoff) all the way through Friday night
    - Saturday: All day
    - Sunday: Until 19:00 NY time (7:00 PM NY time, 2h cool-off buffer after 17:00 open)
    """
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    elif dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    ny = get_ny_time(dt_utc)
    weekday_ny = ny.weekday()
    hour_ny = ny.hour

    # 1. Friday: From cutoff (10:00 AM NY) through entire rest of Friday
    if weekday_ny == 4 and hour_ny >= 10:
        return True, f"WEEKEND_HALT: Friday after 10:00 AM NY cutoff ({ny.strftime('%H:%M %Z')})"

    # 2. Saturday: All day
    if weekday_ny == 5:
        return True, "WEEKEND_HALT: Saturday (Market closed for weekend)"

    # 3. Sunday: Until 19:00 NY time (2h cool-off buffer after 17:00 open)
    if weekday_ny == 6 and hour_ny < 19:
        return True, f"WEEKEND_HALT: Sunday before 19:00 NY open ({ny.strftime('%H:%M %Z')}, 2h cool-off buffer)"

    return False, "OK"

def get_market_status(dt_utc: datetime = None) -> Dict[str, Any]:
    """Return comprehensive market hours and trading status."""
    ny = get_ny_time(dt_utc)
    entry_allowed, entry_reason = is_friday_trade_entry_allowed(dt_utc)
    exit_triggered = is_friday_auto_exit_time(dt_utc)
    weekend_halt, weekend_reason = is_weekend_halt(dt_utc)

    return {
        "ny_time": ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "is_friday": ny.weekday() == 4,
        "is_weekend_halt": weekend_halt,
        "trade_entry_allowed": entry_allowed and not weekend_halt,
        "friday_auto_exit_triggered": exit_triggered,
        "reason": weekend_reason if weekend_halt else entry_reason
    }
