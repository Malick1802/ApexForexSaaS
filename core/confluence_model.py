"""
core/confluence_model.py
Confluence Day/Swing Breakout & Reclamation Model ("Magic Candle Sniper").

Automated Trading Strategy Rules:
1. Day Lines (9:00 PM UTC = 21:00 UTC rollover):
   At the end of each trading day (21:00 UTC), compute the High wick and Low wick
   of the just-closed 24-hour trading day:
   - upper_day_line: Maximum High of M15 candles in [Day_Start, Day_End)
   - lower_day_line: Minimum Low of M15 candles in [Day_Start, Day_End)

2. Swing Lines:
   Looking backwards on M15 candles prior to the Day Line / Day Start:
   - upper_swing_line: The first M15 swing high (local peak) going backwards that is HIGHER than upper_day_line.
   - lower_swing_line: The first M15 swing low (local trough) going backwards that is LOWER than lower_day_line.

3. Continuous M15 Monitoring across major currency pairs.

4. BUY Setup (Bullish Liquidity Sweep & Reclamation):
   - Candle 1: A 15m sell/bearish body candle (Close < Open) must cross either one of the lower lines
     (lower_day_line or lower_swing_line) or both bearish (Open >= line and Close < line).
   - Candle 2 ("Magic Candle"): The next 15m candle must be bullish (Close > Open) and its body
     must cross the same line(s) BULLISH (reclamation: Open <= line and Close > line).
   - Candle 3 (Entry Candle): Enters the moment price meets/touches the High wick of the bullish Magic Candle.
   - SL: Set at Low wick of Magic Candle minus 2.5 pips buffer by default.
   - TP: Entry + RRR * (Entry - SL) with default RRR = 1.5.
   - Expiry: End of Candle 3 (15-minute window).

5. SELL Setup (Bearish Liquidity Sweep & Reclamation):
   - Candle 1: A 15m buy/bullish body candle (Close > Open) must cross either one of the higher lines
     (upper_day_line or upper_swing_line) or both bullish (Open <= line and Close > line).
   - Candle 2 ("Magic Candle"): The next 15m candle must be bearish (Close < Open) and its body
     must cross the same line(s) BEARISH (rejection: Open >= line and Close < line).
   - Candle 3 (Entry Candle): Enters the moment price meets/touches the Low wick of the bearish Magic Candle.
   - SL: Set at High wick of Magic Candle plus 2.5 pips buffer by default.
   - TP: Entry - RRR * (SL - Entry) with default RRR = 1.5.
   - Expiry: End of Candle 3 (15-minute window).

6. Automated Execution:
   - Pending Stop Orders (BUY_STOP / SELL_STOP) placed on MT5 Master account.
   - Broadcast to all authorized copy-trading accounts via execute_signal_for_all_users().
   - Rich Telegram notifications sent upon arming and completion.
"""

import os
import sys
import logging
import threading
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.manual_model import (
    get_mt5,
    get_broker_offset_hours,
    get_pip_size,
    calculate_lot_size,
    submit_manual_order,
)

logger = logging.getLogger("ConfluenceModel")

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFLUENCE_STATE_FILE = PROJECT_ROOT / "logs" / "confluence_state.json"

DEFAULT_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
]


# Asset-Adapted Stop Loss Buffers (in pips)
# Standard Forex uses 2.5 pips. High-volatility / large-pip assets (Gold, Silver, Oil, Crypto)
# are adapted to provide equivalent proportional chart clearance (~20% of M15 candle range)
# and ensure Stop Loss is never placed inside broker spread.
DEFAULT_ASSET_BUFFERS: Dict[str, float] = {
    "forex": 2.5,        # Standard Forex majors and crosses (EURUSD, GBPUSD, USDJPY, etc.)
    "gold": 25.0,        # XAUUSD, GOLD (25.0 pips = $2.50 in 0.1 pip size)
    "silver": 20.0,      # XAGUSD, SILVER (20.0 pips = $0.20 in 0.01 pip size)
    "oil": 15.0,         # USOIL.cash, UKOIL, BRENT (15.0 pips = $0.15 in 0.01 pip size)
    "btc": 60.0,         # BTCUSD (60.0 pips = $60.00 in 1.0 pip size)
    "eth": 30.0,         # ETHUSD (30.0 pips = $3.00 in 0.1 pip size)
    "sol": 25.0,         # SOLUSD (25.0 pips = $0.25 in 0.01 pip size)
    "crypto": 30.0,      # General crypto fallback
    "indices": 25.0,     # US30, NAS100, SPX500, GER40
}

# Asset-Adapted Break-Even Offsets (in pips) for P60 and P25 models
# Ensures the BE adjustment comfortably compensates for broker spread + leaves genuine profit.
DEFAULT_ASSET_BE_OFFSETS: Dict[str, float] = {
    "forex": 2.0,        # Standard Forex (covers 1.0-1.5p spread)
    "gold": 10.0,        # XAUUSD (10.0 pips = $1.00, comfortably clears $0.30 spread)
    "silver": 10.0,      # XAGUSD (10.0 pips = $0.10)
    "oil": 8.0,          # USOIL.cash (8.0 pips = $0.08)
    "btc": 30.0,         # BTCUSD (30.0 pips = $30.00)
    "eth": 15.0,         # ETHUSD (15.0 pips = $1.50)
    "sol": 10.0,         # SOLUSD (10.0 pips = $0.10)
    "crypto": 15.0,      # General crypto fallback
    "indices": 15.0,     # Indices
}

MODEL_MAGIC_MAP: Dict[str, int] = {
    "confluence_ml_p60": 202460,
    "confluence_std_p25": 202425,
    "confluence_ml_m15": 202415,
    "confluence_m15": 202404,
    "manual_m15": 202401,
}

ALL_APEX_MAGICS: Tuple[int, ...] = (202425, 202404, 202460, 202415, 202401)


def is_order_from_model(comment: str, target_model: str) -> bool:
    """
    Check if an MT5 order or position comment belongs to the specified model.
    Enables strict deduplication for the SAME model while permitting DIFFERENT models
    to independently place their authorized setups on the same currency pair.
    """
    comm = str(comment or "").upper()
    mod = str(target_model or "").lower()

    if mod == "confluence_ml_p60":
        return "P60" in comm or "ML-P60" in comm
    elif mod == "confluence_std_p25":
        return "P25" in comm or "STD-P25" in comm
    elif mod in ("confluence_ml_m15", "confluence_ml"):
        return ("APEX-ML" in comm or "CONFLUENCE-ML" in comm) and "P60" not in comm
    elif mod in ("confluence_m15", "confluence_std"):
        return ("APEX-STD" in comm or "CONFLUENCE" in comm) and "P25" not in comm and "P60" not in comm
    elif mod == "manual_m15":
        return ("APEX-M15" in comm or "MANUAL" in comm) and not any(k in comm for k in ("P60", "P25", "APEX-ML", "APEX-STD"))
    return False



def get_symbol_sl_buffer(
    symbol: str,
    cfg: Optional[Dict[str, Any]] = None,
    override_buffer: Optional[float] = None
) -> float:
    """
    Return the adapted Stop Loss buffer in pips for a symbol based on its asset class and pip scale.
    Guarantees that large-pip assets (Gold, Silver, Oil, Crypto) have appropriate wick clearance
    rather than applying a tiny Forex buffer. Also enforces broker spread floor.
    """
    if cfg is None:
        cfg = load_confluence_config()

    s = symbol.upper().replace(".CASH", "").replace(".M", "").replace(".RAW", "").replace("_", "")

    # 1. Direct symbol override in config takes absolute precedence
    symbol_buffers = cfg.get("symbol_buffers", {})
    if symbol in symbol_buffers:
        base_buffer = float(symbol_buffers[symbol])
    elif s in symbol_buffers:
        base_buffer = float(symbol_buffers[s])
    # 2. Check if a non-default custom override was passed by caller
    elif override_buffer is not None and override_buffer != 2.5:
        base_buffer = float(override_buffer)
    else:
        # 3. Resolve by asset class
        asset_bufs = cfg.get("asset_buffers", DEFAULT_ASSET_BUFFERS)
        if any(x in s for x in ("XAU", "GOLD")):
            base_buffer = float(asset_bufs.get("gold", 25.0))
        elif any(x in s for x in ("XAG", "SILVER")):
            base_buffer = float(asset_bufs.get("silver", 20.0))
        elif any(x in s for x in ("OIL", "WTI", "BRENT", "USOIL", "UKOIL")):
            base_buffer = float(asset_bufs.get("oil", 15.0))
        elif "BTC" in s:
            base_buffer = float(asset_bufs.get("btc", 60.0))
        elif "ETH" in s:
            base_buffer = float(asset_bufs.get("eth", 30.0))
        elif "SOL" in s:
            base_buffer = float(asset_bufs.get("sol", 25.0))
        elif any(x in s for x in ("US30", "NAS100", "SPX", "GER", "DAX")):
            base_buffer = float(asset_bufs.get("indices", 25.0))
        else:
            base_buffer = float(cfg.get("sl_buffer_pips", asset_bufs.get("forex", 2.5)))

    # 4. Spread Floor Protection: Buffer must be >= 1.5x live spread to prevent instant stop-out
    try:
        mt5 = get_mt5()
        if mt5:
            tick = mt5.symbol_info_tick(symbol)
            pip_sz = get_pip_size(symbol)
            if tick and tick.ask > tick.bid and pip_sz > 0:
                spread_pips = (tick.ask - tick.bid) / pip_sz
                if spread_pips > 0:
                    base_buffer = max(base_buffer, round(spread_pips * 1.5, 1))
    except Exception:
        pass

    return round(base_buffer, 1)


def get_symbol_be_offset(symbol: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Return the spread-compensating Break-Even offset in pips for a symbol.
    Ensures that when moving SL to breakeven, the offset comfortably covers broker spread.
    """
    if cfg is None:
        cfg = load_confluence_config()

    s = symbol.upper().replace(".CASH", "").replace(".M", "").replace(".RAW", "").replace("_", "")

    symbol_be = cfg.get("symbol_be_offsets", {})
    if symbol in symbol_be:
        be_offset = float(symbol_be[symbol])
    elif s in symbol_be:
        be_offset = float(symbol_be[s])
    else:
        asset_be = cfg.get("asset_be_offsets", DEFAULT_ASSET_BE_OFFSETS)
        if any(x in s for x in ("XAU", "GOLD")):
            be_offset = float(asset_be.get("gold", 10.0))
        elif any(x in s for x in ("XAG", "SILVER")):
            be_offset = float(asset_be.get("silver", 10.0))
        elif any(x in s for x in ("OIL", "WTI", "BRENT", "USOIL", "UKOIL")):
            be_offset = float(asset_be.get("oil", 8.0))
        elif "BTC" in s:
            be_offset = float(asset_be.get("btc", 30.0))
        elif "ETH" in s:
            be_offset = float(asset_be.get("eth", 15.0))
        elif "SOL" in s:
            be_offset = float(asset_be.get("sol", 10.0))
        elif any(x in s for x in ("US30", "NAS100", "SPX", "GER", "DAX")):
            be_offset = float(asset_be.get("indices", 15.0))
        else:
            be_offset = float(asset_be.get("forex", 2.0))

    # Spread Floor Protection: BE offset must be >= spread_pips + 1.0
    try:
        mt5 = get_mt5()
        if mt5:
            tick = mt5.symbol_info_tick(symbol)
            pip_sz = get_pip_size(symbol)
            if tick and tick.ask > tick.bid and pip_sz > 0:
                spread_pips = (tick.ask - tick.bid) / pip_sz
                if spread_pips > 0:
                    be_offset = max(be_offset, round(spread_pips + 1.0, 1))
    except Exception:
        pass

    return round(be_offset, 1)


def load_confluence_config() -> Dict[str, Any]:
    """Load configuration for the Confluence Model from config.yaml."""
    cfg = {
        "enabled": True,
        "auto_trade": True,
        "enable_ml_model": True,
        "enable_standard_model": False,
        "sl_buffer_pips": 2.5,
        "asset_buffers": dict(DEFAULT_ASSET_BUFFERS),
        "asset_be_offsets": dict(DEFAULT_ASSET_BE_OFFSETS),
        "rrr": 1.5,
        "risk_type": "percent",
        "risk_value": 0.5,
        "ml_filter_enabled": True,
        "ml_threshold": 0.48,
        "include_wicks": True,
        "symbols": DEFAULT_SYMBOLS,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                if "confluence_model" in raw:
                    cfg.update(raw["confluence_model"])
        except Exception as e:
            logger.warning(f"Error loading confluence config: {e}")
    return cfg


def save_confluence_config(cfg: Dict[str, Any]) -> bool:
    """Save Confluence Model configuration to config.yaml."""
    try:
        raw = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        raw["confluence_model"] = cfg
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Error saving confluence config: {e}")
        return False


def get_day_lines(symbol: str, as_of_utc: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """
    Compute Day Lines for a symbol as of a given UTC timestamp.
    Trading day closes at 21:00 UTC (9pm UTC).
    Returns:
      {
        'upper_day_line': float,
        'lower_day_line': float,
        'day_start_utc': datetime,
        'day_end_utc': datetime,
        'day_high_time': datetime,
        'day_low_time': datetime,
        'bar_count': int
      }
    """
    mt5 = get_mt5()
    if not mt5:
        return None

    if as_of_utc is None:
        as_of_utc = datetime.now(timezone.utc)

    offset_h = get_broker_offset_hours(symbol)

    # Determine the target closed 24-hour day boundary ending at 21:00 UTC
    if as_of_utc.hour < 21:
        day_end = datetime(as_of_utc.year, as_of_utc.month, as_of_utc.day, 21, 0, tzinfo=timezone.utc) - timedelta(days=1)
    else:
        day_end = datetime(as_of_utc.year, as_of_utc.month, as_of_utc.day, 21, 0, tzinfo=timezone.utc)

    # Look back up to 5 days to find the most recent trading session with bars (handles weekends)
    day_bars = []
    day_start = day_end - timedelta(days=1)

    # Fetch ~500 M15 bars to cover the past week
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 500)
    if rates is None or len(rates) < 20:
        return None

    all_bars = []
    for r in rates:
        raw_dt = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
        utc_dt = raw_dt - timedelta(hours=offset_h)
        # Approach 1 (No Gap): Exclude 21:00 UTC rollover gap / spread blowout candle
        if utc_dt.hour == 21 and utc_dt.minute == 0:
            continue
        all_bars.append({
            "time": utc_dt,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })

    # Search for the most recent completed day window that has bars
    for day_shift in range(5):
        target_end = day_end - timedelta(days=day_shift)
        target_start = target_end - timedelta(days=1)
        bars_in_window = [b for b in all_bars if target_start <= b["time"] < target_end]
        if len(bars_in_window) >= 15: # Valid trading day found
            day_bars = bars_in_window
            day_start = target_start
            day_end = target_end
            break

    if not day_bars:
        return None

    upper_day_line = max(b["high"] for b in day_bars)
    lower_day_line = min(b["low"] for b in day_bars)

    day_high_bar = next(b for b in day_bars if b["high"] == upper_day_line)
    day_low_bar = next(b for b in day_bars if b["low"] == lower_day_line)

    return {
        "upper_day_line": round(upper_day_line, 5),
        "lower_day_line": round(lower_day_line, 5),
        "day_start_utc": day_start,
        "day_end_utc": day_end,
        "day_high_time": day_high_bar["time"],
        "day_low_time": day_low_bar["time"],
        "bar_count": len(day_bars),
    }


def get_swing_lines(
    symbol: str,
    upper_day_line: float,
    lower_day_line: float,
    prior_to_utc: datetime,
    lookback_bars: int = 1500,
) -> Dict[str, Any]:
    """
    Search backwards on M15 candles prior to `prior_to_utc`:
    - upper_swing_line: The first M15 swing high going backwards that is HIGHER than upper_day_line.
    - lower_swing_line: The first M15 swing low going backwards that is LOWER than lower_day_line.
    Filters out 21:00 UTC rollover gap / spread spike bars to ensure clean price action swings.
    """
    mt5 = get_mt5()
    result = {
        "upper_swing_line": None,
        "lower_swing_line": None,
        "upper_swing_time": None,
        "lower_swing_time": None,
    }
    if not mt5:
        return result

    offset_h = get_broker_offset_hours(symbol)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, lookback_bars)
    if rates is None or len(rates) < 50:
        return result

    bars = []
    for r in rates:
        raw_dt = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
        utc_dt = raw_dt - timedelta(hours=offset_h)
        # Approach 1 (No Gap): Exclude 21:00 UTC rollover gap / spread blowout candle
        if utc_dt.hour == 21 and utc_dt.minute == 0:
            continue
        if utc_dt < prior_to_utc:
            bars.append({
                "time": utc_dt,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            })

    if len(bars) < 5:
        return result

    pip_sz = get_pip_size(symbol)
    min_clearance = 2.5 * pip_sz

    # Scan backwards from the most recent prior bar
    upper_found = False
    lower_found = False

    fallback_upper = None
    fallback_upper_time = None
    fallback_lower = None
    fallback_lower_time = None

    for i in range(len(bars) - 2, 1, -1):
        b = bars[i]
        b_prev = bars[i - 1]
        b_next = bars[i + 1]

        # Check Swing High (local peak)
        if not upper_found and b["high"] > b_prev["high"] and b["high"] > b_next["high"]:
            if b["high"] > upper_day_line:
                if fallback_upper is None:
                    fallback_upper = round(b["high"], 5)
                    fallback_upper_time = b["time"]
                # Must clear day line by at least 2.5 pips to qualify as a structural swing high
                if b["high"] >= upper_day_line + min_clearance:
                    result["upper_swing_line"] = round(b["high"], 5)
                    result["upper_swing_time"] = b["time"]
                    upper_found = True

        # Check Swing Low (local trough)
        if not lower_found and b["low"] < b_prev["low"] and b["low"] < b_next["low"]:
            if b["low"] < lower_day_line:
                if fallback_lower is None:
                    fallback_lower = round(b["low"], 5)
                    fallback_lower_time = b["time"]
                # Must clear day line by at least 2.5 pips to qualify as a structural swing low
                if b["low"] <= lower_day_line - min_clearance:
                    result["lower_swing_line"] = round(b["low"], 5)
                    result["lower_swing_time"] = b["time"]
                    lower_found = True

        if upper_found and lower_found:
            break

    # If no swing cleared the min buffer, fall back to the first peak/trough outside the day lines
    if not upper_found and fallback_upper is not None:
        result["upper_swing_line"] = fallback_upper
        result["upper_swing_time"] = fallback_upper_time
    if not lower_found and fallback_lower is not None:
        result["lower_swing_line"] = fallback_lower
        result["lower_swing_time"] = fallback_lower_time

    return result


def evaluate_confluence_setup(
    symbol: str,
    rrr: float = 1.5,
    sl_buffer_pips: Optional[float] = None,
    risk_type: str = "percent",
    risk_value: float = 0.5,
    active_model: Optional[str] = None,
    include_wicks: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Inspect the recent M15 candles for an active Confluence Breakout & Reclamation setup.
    - Candle 1 = rates[-3] (Breakout candle)
    - Candle 2 = rates[-2] (Magic candle - just closed)
    - Candle 3 = rates[-1] (Current forming / entry candle)

    Returns a complete evaluation dictionary including setup status and order parameters.
    """
    cfg = load_confluence_config()
    actual_buffer_pips = get_symbol_sl_buffer(symbol, cfg, override_buffer=sl_buffer_pips)

    res = {
        "valid": False,
        "symbol": symbol,
        "setup_detected": False,
        "direction": None,
        "line_type": None,
        "reclaimed_line": None,
        "day_lines": None,
        "swing_lines": None,
        "candle_1": None,
        "magic_candle": None,
        "entry_candle": None,
        "entry": None,
        "sl": None,
        "tp": None,
        "rrr": rrr,
        "sl_pips": 0.0,
        "tp_pips": 0.0,
        "sl_buffer_pips": actual_buffer_pips,
        "lots": 0.01,
        "order_type": None,
        "expiry_utc": None,
        "error": None,
    }

    mt5 = get_mt5()
    if not mt5:
        res["error"] = "MT5 not connected."
        return res

    offset_h = get_broker_offset_hours(symbol)
    now_utc = datetime.now(timezone.utc)

    # 1. Day lines and Swing lines
    day_info = get_day_lines(symbol, as_of_utc=now_utc)
    if not day_info:
        res["error"] = "Could not compute Day Lines (insufficient historical data)."
        return res

    res["day_lines"] = day_info
    upper_day = day_info["upper_day_line"]
    lower_day = day_info["lower_day_line"]

    swing_info = get_swing_lines(
        symbol=symbol,
        upper_day_line=upper_day,
        lower_day_line=lower_day,
        prior_to_utc=day_info["day_start_utc"],
    )
    res["swing_lines"] = swing_info

    upper_swing = swing_info["upper_swing_line"]
    lower_swing = swing_info["lower_swing_line"]

    # 2. Fetch the last 10 M15 candles
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 10)
    if rates is None or len(rates) < 4:
        res["error"] = "Could not fetch recent M15 bars."
        return res

    pip_sz = get_pip_size(symbol)
    buffer_amt = actual_buffer_pips * pip_sz

    def _parse_bar(r):
        raw_dt = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
        open_utc = raw_dt - timedelta(hours=offset_h)
        return {
            "time": open_utc,
            "close_time": open_utc + timedelta(minutes=15),
            "open": round(float(r["open"]), 5),
            "high": round(float(r["high"]), 5),
            "low": round(float(r["low"]), 5),
            "close": round(float(r["close"]), 5),
            "is_bullish": float(r["close"]) > float(r["open"]),
            "is_bearish": float(r["close"]) < float(r["open"]),
        }

    c_entry = _parse_bar(rates[-1]) # Candle 3 (Current forming bar)
    c_magic = _parse_bar(rates[-2]) # Candle 2 (Magic candle - just closed)
    c_break = _parse_bar(rates[-3]) # Candle 1 (Breakout candle)

    res["candle_1"] = c_break
    res["magic_candle"] = c_magic
    res["entry_candle"] = c_entry

    # High lines to test for SELL setups
    high_lines = []
    if upper_day is not None:
        high_lines.append(("Upper Day Line", upper_day))
    if upper_swing is not None:
        high_lines.append(("Upper Swing Line", upper_swing))

    # Low lines to test for BUY setups
    low_lines = []
    if lower_day is not None:
        low_lines.append(("Lower Day Line", lower_day))
    if lower_swing is not None:
        low_lines.append(("Lower Swing Line", lower_swing))

    cfg = load_confluence_config()
    if include_wicks is None:
        include_wicks = bool(cfg.get("include_wicks", True))

    # -------------------------------------------------------------
    # Check BUY SETUP:
    # Liquidity sweep across lower Day Line or Swing Line + Reclamation:
    # Candle 1: bearish body/sweep penetrating across the lower line (body or wick)
    # Candle 2 (Magic Candle): bullish body/sweep reclaiming above the line
    # -------------------------------------------------------------
    if c_break["is_bearish"] and c_magic["is_bullish"]:
        for name, line_val in low_lines:
            if include_wicks:
                # Candle 1 sweeps line: either body crossed below, or lower wick pierced below line
                c1_sweep = (
                    (c_break["open"] >= line_val and c_break["close"] < line_val) or
                    (c_break["low"] <= line_val and max(c_break["open"], c_break["high"]) >= line_val) or
                    (c_break["close"] < line_val)
                )
                # Candle 2 reclaims line: touched or came from at/below line, and closed back above line
                c2_reclaim = (
                    (min(c_magic["low"], c_magic["open"]) <= line_val and c_magic["close"] > line_val) or
                    (c_magic["open"] <= line_val and c_magic["close"] > line_val)
                )
                valid_buy = c1_sweep and c2_reclaim
            else:
                c1_bearish_cross = (c_break["open"] >= line_val and c_break["close"] < line_val)
                c2_bullish_cross = (c_magic["open"] <= line_val and c_magic["close"] > line_val)
                valid_buy = c1_bearish_cross and c2_bullish_cross

            if valid_buy:
                entry = round(c_magic["high"], 5)
                sl = round(c_magic["low"] - buffer_amt, 5)
                sl_dist = abs(entry - sl)
                tp_dist = sl_dist * rrr
                tp = round(entry + tp_dist, 5)

                lot_calc = calculate_lot_size(symbol, entry, sl, risk_type, risk_value)

                res["valid"] = True
                res["setup_detected"] = True
                res["direction"] = "BUY"
                res["order_type"] = "BUY_STOP"
                res["line_type"] = name
                res["reclaimed_line"] = line_val
                res["entry"] = entry
                res["sl"] = sl
                res["tp"] = tp
                res["sl_pips"] = round(sl_dist / pip_sz, 1)
                res["tp_pips"] = round(tp_dist / pip_sz, 1)
                res["lots"] = lot_calc.get("lots", 0.01)
                res["risk_usd"] = lot_calc.get("risk_usd", 0.0)
                res["balance"] = lot_calc.get("balance", 0.0)
                res["expiry_utc"] = c_entry["close_time"]
                break

    # -------------------------------------------------------------
    # Check SELL SETUP:
    # Liquidity sweep across upper Day Line or Swing Line + Reclamation:
    # Candle 1: bullish body/sweep penetrating across the upper line (body or wick)
    # Candle 2 (Magic Candle): bearish body/sweep reclaiming below the line
    # -------------------------------------------------------------
    if not res.get("setup_detected") and c_break["is_bullish"] and c_magic["is_bearish"]:
        for name, line_val in high_lines:
            if include_wicks:
                # Candle 1 sweeps line: either body crossed above, or upper wick pierced above line
                c1_sweep = (
                    (c_break["open"] <= line_val and c_break["close"] > line_val) or
                    (c_break["high"] >= line_val and min(c_break["open"], c_break["low"]) <= line_val) or
                    (c_break["close"] > line_val)
                )
                # Candle 2 reclaims line: touched or came from at/above line, and closed back below line
                c2_reclaim = (
                    (max(c_magic["high"], c_magic["open"]) >= line_val and c_magic["close"] < line_val) or
                    (c_magic["open"] >= line_val and c_magic["close"] < line_val)
                )
                valid_sell = c1_sweep and c2_reclaim
            else:
                c1_bullish_cross = (c_break["open"] <= line_val and c_break["close"] > line_val)
                c2_bearish_cross = (c_magic["open"] >= line_val and c_magic["close"] < line_val)
                valid_sell = c1_bullish_cross and c2_bearish_cross

            if valid_sell:
                entry = round(c_magic["low"], 5)
                sl = round(c_magic["high"] + buffer_amt, 5)
                sl_dist = abs(sl - entry)
                tp_dist = sl_dist * rrr
                tp = round(entry - tp_dist, 5)

                lot_calc = calculate_lot_size(symbol, entry, sl, risk_type, risk_value)

                res["valid"] = True
                res["setup_detected"] = True
                res["direction"] = "SELL"
                res["order_type"] = "SELL_STOP"
                res["line_type"] = name
                res["reclaimed_line"] = line_val
                res["entry"] = entry
                res["sl"] = sl
                res["tp"] = tp
                res["sl_pips"] = round(sl_dist / pip_sz, 1)
                res["tp_pips"] = round(tp_dist / pip_sz, 1)
                res["lots"] = lot_calc.get("lots", 0.01)
                res["risk_usd"] = lot_calc.get("risk_usd", 0.0)
                res["balance"] = lot_calc.get("balance", 0.0)
                res["expiry_utc"] = c_entry["close_time"]
                break

    cfg = load_confluence_config()
    target_model = active_model or cfg.get("active_model", "confluence_ml_p60")
    res["model_version"] = target_model

    # Model Configuration & Partial Profit Assignment:
    # - New Model 1 (confluence_ml_p60): ML Gate, 60% partial TP, BE+2p SL move, concurrent asset trading
    # - New Model 2 (confluence_std_p25): Standard Rule-Based, 25% partial TP, BE+2p SL move, concurrent asset trading
    # - Original Model A (confluence_ml_m15): Original ML Gate, NO partial TP (runs full trade to TP/SL), standard non-concurrent
    # - Original Model B (confluence_m15): Original Rule-Based, NO partial TP (runs full trade to TP/SL), standard non-concurrent
    if target_model == "confluence_ml_p60":
        partial_ratio = 0.60
        be_offset_pips = get_symbol_be_offset(symbol, cfg)
    elif target_model == "confluence_std_p25":
        partial_ratio = 0.25
        be_offset_pips = get_symbol_be_offset(symbol, cfg)
    else:
        # Original models (confluence_ml_m15, confluence_m15) have NO partial profit or BE adjustment
        partial_ratio = None
        be_offset_pips = 0.0

    res["partial_target_ratio"] = partial_ratio
    res["be_offset_pips"] = be_offset_pips
    res["partial_target_price"] = None
    res["be_sl_price"] = None

    if res.get("setup_detected") and res.get("entry") is not None and res.get("tp") is not None:
        if partial_ratio is not None:
            tp_dist = abs(res["tp"] - res["entry"])
            if res["direction"] == "BUY":
                res["partial_target_price"] = round(res["entry"] + (partial_ratio * tp_dist), 5)
                res["be_sl_price"] = round(res["entry"] + (be_offset_pips * pip_sz), 5)
            else:
                res["partial_target_price"] = round(res["entry"] - (partial_ratio * tp_dist), 5)
                res["be_sl_price"] = round(res["entry"] - (be_offset_pips * pip_sz), 5)

    # Run LightGBM Meta-Labeling Quality Evaluation if setup detected
    if res.get("setup_detected"):
        is_ml_gate = target_model in ("confluence_ml_p60", "confluence_ml_m15")
        if is_ml_gate:
            try:
                from core.ml_filter import get_ml_filter
                ml_th = float(cfg.get("ml_threshold", 0.48))
                res["ml_filter"] = get_ml_filter(default_threshold=ml_th).evaluate_setup(res, mt5_inst=mt5)
            except Exception as _me:
                logger.warning(f"ML filter setup evaluation failed: {_me}")
                res["ml_filter"] = {"evaluated": False, "passed": True, "probability": 0.50, "recommendation": "BYPASS"}
        else:
            # Standard Rule-Based Model: ML gate bypassed
            res["ml_filter"] = {
                "evaluated": False,
                "passed": True,
                "probability": 0.50,
                "confidence_score_pct": 50.0,
                "threshold": 0.48,
                "recommendation": "BYPASS",
                "reason": "Standard Rule-Based Confluence Model (ML Gate Bypassed)"
            }

    res["valid"] = True
    return res


_confluence_lock = threading.Lock()
_spent_candle_setups: set = set()


def _load_confluence_state() -> Dict[str, Any]:
    """Load persisted Confluence Model execution state (deduplication history)."""
    with _confluence_lock:
        if CONFLUENCE_STATE_FILE.exists():
            try:
                with open(CONFLUENCE_STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"placed_setups": {}}


def _save_confluence_state(state: Dict[str, Any]) -> None:
    """Save Confluence Model state atomically."""
    with _confluence_lock:
        try:
            CONFLUENCE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = CONFLUENCE_STATE_FILE.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            import os
            os.replace(tmp_file, CONFLUENCE_STATE_FILE)
        except Exception as e:
            logger.warning(f"Could not save confluence state: {e}")


def execute_confluence_setup(
    setup: Dict[str, Any],
    broadcast_to_subscribers: bool = True,
    send_telegram: bool = True,
) -> Dict[str, Any]:
    """
    Execute a detected Confluence Setup automatically:
    1. Multi-Tier Deduplication:
       - Process in-memory spent candle guard
       - Persistent SQLite signals.db verification
       - Atomic JSON state check
       - MT5 Broker-level pending order and active candle position check
    2. Submit pending stop order to MT5 Master.
    3. Broadcast to all enabled copy-trading follower accounts.
    4. Record to SignalDatabase and send rich Telegram alert.
    """
    res = {"success": False, "ticket": None, "error": None}
    if not setup.get("setup_detected") or not setup.get("valid"):
        res["error"] = setup.get("error", "No valid setup to execute.")
        return res

    symbol = setup["symbol"]
    direction = setup["direction"]
    target_model = setup.get("model_version", "confluence_ml_p60")

    # UNIFIED MODEL GATEKEEPER: Ensure model is explicitly authorized for LIVE execution
    from core.model_gatekeeper import is_model_live_authorized
    if not is_model_live_authorized(target_model):
        res["error"] = f"Model '{target_model}' is in SHADOW mode (not selected for live execution). Blocking MT5 order."
        logger.info(f"👻 MODEL GATEKEEPER: {target_model} is in SHADOW mode. Bypassing live MT5 execution for {symbol} {direction}.")
        return res

    # DYNAMIC YTD WINNING ASSET GATE: Ensure symbol is a YTD winning asset for this model
    from core.dynamic_model_whitelist import is_pair_whitelisted_for_model
    if not is_pair_whitelisted_for_model(target_model, symbol):
        res["error"] = f"Symbol '{symbol}' is benched under {target_model} Dynamic YTD Whitelist (Net R < 0.0)."
        logger.info(f"🛡️ DYNAMIC YTD GATE: {res['error']}. Bypassing live MT5 execution.")
        return res

    magic_candle_dt = setup.get("magic_candle", {}).get("time")
    if hasattr(magic_candle_dt, "strftime"):
        magic_time_iso = magic_candle_dt.strftime("%Y-%m-%d %H:%M")
        magic_time_str = magic_candle_dt.isoformat()
    else:
        magic_time_iso = str(magic_candle_dt)[:16]
        magic_time_str = str(magic_candle_dt)

    dedup_key = f"{target_model}:{symbol}:{direction}:{magic_time_str}"
    dedup_key_norm = f"{target_model}:{symbol}:{direction}:{magic_time_iso}"

    # ── TIER 1: In-Memory Process-Level Spent Candle Cache ─────────────
    with _confluence_lock:
        if dedup_key in _spent_candle_setups or dedup_key_norm in _spent_candle_setups:
            res["error"] = f"Setup already executed in memory for {symbol} on candle {magic_time_iso} by {target_model}."
            logger.info(f"⏭️ Skipping duplicate confluence execution (in-memory spent): {dedup_key_norm}")
            return res

    # ── TIER 2: Database Persistent Deduplication (Source of Truth) ─────
    try:
        from core.database import SignalDatabase
        db = SignalDatabase()
        with db._get_connection() as conn:
            cur = conn.cursor()
            cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
            cur.execute("""
                SELECT id, mt5_ticket, timestamp, outcome FROM signals 
                WHERE symbol = ? AND signal = ? AND model_version = ?
                  AND timestamp >= ?
                LIMIT 1
            """, (symbol, direction, target_model, cutoff_iso))
            existing_row = cur.fetchone()
            if existing_row:
                with _confluence_lock:
                    _spent_candle_setups.add(dedup_key)
                    _spent_candle_setups.add(dedup_key_norm)
                res["error"] = f"Setup already recorded in DB for {symbol} on candle {magic_time_iso} by {target_model} (Signal ID #{existing_row[0]}). Duplicate prevented."
                logger.info(f"🛑 DB DEDUP: {res['error']}")
                return res
    except Exception as dbe:
        logger.warning(f"DB dedup check warning: {dbe}")

    # ── TIER 3: JSON File State Deduplication ──────────────────────────
    state = _load_confluence_state()
    placed_setups = state.get("placed_setups", {})
    if dedup_key in placed_setups or dedup_key_norm in placed_setups:
        prev = placed_setups.get(dedup_key) or placed_setups.get(dedup_key_norm)
        with _confluence_lock:
            _spent_candle_setups.add(dedup_key)
            _spent_candle_setups.add(dedup_key_norm)
        res["error"] = f"Setup already executed for {symbol} on candle {magic_time_iso} by {target_model} (Ticket #{prev.get('ticket')})."
        logger.info(f"⏭️ Skipping duplicate confluence execution: {dedup_key_norm}")
        return res

    cfg = load_confluence_config()
    # Concurrency rule: ONLY confluence_ml_p60 and confluence_std_p25 are allowed concurrent asset trades!
    allow_concurrent = (target_model in ("confluence_ml_p60", "confluence_std_p25")) or bool(cfg.get("allow_concurrent_asset", False))

    # ── TIER 4: MT5 Broker-Level Pending Order & Position Deduplication (Per Model) ──
    mt5 = get_mt5()
    if mt5:
        pip_sz = get_pip_size(symbol)
        target_magic = MODEL_MAGIC_MAP.get(target_model, 202425)
        existing_orders = mt5.orders_get(symbol=symbol)
        if existing_orders:
            for o in existing_orders:
                comm = str(getattr(o, "comment", "") or "").upper()
                o_magic = getattr(o, "magic", 0)
                # Check if this pending order belongs to the SAME model:
                # E.g. confluence_ml_p60 only checks against P60 orders; confluence_std_p25 only checks P25 orders.
                # DIFFERENT models ARE ALLOWED to place orders on the same symbol!
                is_same_model_order = is_order_from_model(comm, target_model) or (o_magic == target_magic)
                if is_same_model_order:
                    # Same direction pending order check
                    is_same_dir = (direction == "BUY" and o.type == mt5.ORDER_TYPE_BUY_STOP) or (direction == "SELL" and o.type == mt5.ORDER_TYPE_SELL_STOP)
                    if is_same_dir or abs(o.price_open - setup["entry"]) < (3.0 * pip_sz):
                        with _confluence_lock:
                            _spent_candle_setups.add(dedup_key)
                            _spent_candle_setups.add(dedup_key_norm)
                        res["error"] = f"Active MT5 pending order #{o.ticket} already exists for {symbol} ({direction}) by {target_model}. Duplicate prevented."
                        logger.warning(f"🛑 DEDUP: {res['error']}")
                        return res
                    if not allow_concurrent:
                        res["error"] = f"Active MT5 pending order #{o.ticket} already exists for {symbol} by {target_model}. Duplicate prevented."
                        logger.warning(f"🛑 DEDUP: {res['error']}")
                        return res

        existing_pos = mt5.positions_get(symbol=symbol)
        if existing_pos:
            for p in existing_pos:
                comm = str(getattr(p, "comment", "") or "").upper()
                p_magic = getattr(p, "magic", 0)
                # Check if this active position belongs to the SAME model:
                is_same_model_pos = is_order_from_model(comm, target_model) or (p_magic == target_magic)
                if is_same_model_pos:
                    p_time = getattr(p, "time", 0)
                    now_ts = int(datetime.now(timezone.utc).timestamp())
                    # If an existing position was opened during the CURRENT M15 candle (< 15 mins) by the SAME model, BLOCK duplicate!
                    is_current_candle_pos = (now_ts - p_time) < (15 * 60)
                    if not allow_concurrent or is_current_candle_pos:
                        with _confluence_lock:
                            _spent_candle_setups.add(dedup_key)
                            _spent_candle_setups.add(dedup_key_norm)
                        res["error"] = f"Active MT5 position #{p.ticket} already running for {symbol} by {target_model} (opened {now_ts - p_time}s ago). Duplicate prevented."
                        logger.warning(f"🛑 DEDUP: {res['error']}")
                        return res

    # Build comment string
    if target_model == "confluence_ml_p60":
        comment_str = f"APEX-ML-P60 {direction} {setup.get('rrr', 1.5)}R"
    elif target_model == "confluence_std_p25":
        comment_str = f"APEX-STD-P25 {direction} {setup.get('rrr', 1.5)}R"
    elif target_model == "confluence_ml_m15":
        comment_str = f"APEX-ML {direction} {setup.get('rrr', 1.5)}R"
    else:
        comment_str = f"APEX-STD {direction} {setup.get('rrr', 1.5)}R"

    order_spec = {
        "valid": True,
        "error": None,
        "symbol": symbol,
        "direction": direction,
        "entry": setup["entry"],
        "sl": setup["sl"],
        "tp": setup["tp"],
        "rrr": setup["rrr"],
        "sl_pips": setup["sl_pips"],
        "tp_pips": setup["tp_pips"],
        "sl_buffer_pips": setup["sl_buffer_pips"],
        "risk_type": setup.get("risk_type", "percent"),
        "risk_value": setup.get("risk_value", 0.5),
        "lots": setup["lots"],
        "risk_usd": setup.get("risk_usd", 0.0),
        "balance": setup.get("balance", 0.0),
        "order_type": setup["order_type"],
        "expiry_utc": setup["expiry_utc"],
        "magic": MODEL_MAGIC_MAP.get(target_model, 202425),
        "candle": {
            "time": setup["magic_candle"]["time"],
            "high": setup["magic_candle"]["high"],
            "low": setup["magic_candle"]["low"],
            "close": setup["magic_candle"]["close"],
        },
        "model_version": target_model,
        "allow_concurrent_asset": allow_concurrent,
        "comment": comment_str,
        "confluence_info": {
            "line_type": setup["line_type"],
            "reclaimed_line": setup["reclaimed_line"],
        }
    }

    logger.info(f"⚡ EXECUTING AUTOMATED CONFLUENCE SETUP: {symbol} {direction} @ {setup['entry']:.5f} ({target_model} · Line: {setup['line_type']} {setup['reclaimed_line']:.5f})")

    submit_res = submit_manual_order(
        order_spec=order_spec,
        broadcast_to_subscribers=broadcast_to_subscribers,
        send_telegram=False, # We send our custom Confluence Telegram alert below
    )

    if submit_res.get("success"):
        res["success"] = True
        res["ticket"] = submit_res.get("ticket")
        res["broadcast_results"] = submit_res.get("broadcast_results", {})

        # Record in state for deduplication
        with _confluence_lock:
            _spent_candle_setups.add(dedup_key)
            _spent_candle_setups.add(dedup_key_norm)

        placed_setups[dedup_key] = {
            "ticket": submit_res.get("ticket"),
            "symbol": symbol,
            "direction": direction,
            "model_version": target_model,
            "placed_at": datetime.now(timezone.utc).isoformat(),
            "line_type": setup["line_type"],
            "reclaimed_line": setup["reclaimed_line"],
        }
        # Keep only the last 200 placed setups to prevent memory bloat
        if len(placed_setups) > 200:
            keys_to_remove = list(placed_setups.keys())[:-200]
            for k in keys_to_remove:
                placed_setups.pop(k, None)
        state["placed_setups"] = placed_setups

        # Only record in managed_positions if the model uses partial profit execution (confluence_ml_p60, confluence_std_p25)
        if setup.get("partial_target_ratio") is not None:
            managed_positions = state.setdefault("managed_positions", {})
            managed_positions[str(submit_res.get("ticket"))] = {
                "ticket": submit_res.get("ticket"),
                "symbol": symbol,
                "direction": direction,
                "model_version": target_model,
                "entry": setup["entry"],
                "sl": setup["sl"],
                "tp": setup["tp"],
                "partial_target_ratio": setup.get("partial_target_ratio"),
                "partial_target_price": setup.get("partial_target_price"),
                "be_offset_pips": setup.get("be_offset_pips", 2.0),
                "be_sl_price": setup.get("be_sl_price"),
                "partial_taken": False,
                "be_sl_moved": False,
                "placed_at": datetime.now(timezone.utc).isoformat(),
            }
            state["managed_positions"] = managed_positions

        _save_confluence_state(state)

        # Rich Confluence Telegram Notification
        if send_telegram:
            try:
                _send_confluence_telegram_alert(setup, submit_res)
            except Exception as e:
                logger.warning(f"Confluence Telegram alert failed: {e}")
    else:
        res["error"] = submit_res.get("error", "Order placement failed.")
        logger.error(f"❌ Confluence order submission failed for {symbol}: {res['error']}")
        # Mark spent in memory so failed order attempts don't spam MT5 every minute on the same candle
        with _confluence_lock:
            _spent_candle_setups.add(dedup_key)
            _spent_candle_setups.add(dedup_key_norm)

    return res


def record_confluence_shadow_trade(
    setup: Dict[str, Any],
    model_version: str = "confluence_m15",
    is_suppressed: bool = False,
    ml_score: Optional[float] = None,
) -> Optional[int]:
    """
    Record a non-active or ML-suppressed Confluence setup as a background SHADOW trade in signals.db.
    Allows tracking and comparing model outcomes in real-time without risking live broker capital.
    """
    if not setup.get("setup_detected") or not setup.get("valid"):
        return None

    symbol = setup["symbol"]
    direction = setup["direction"]
    magic_candle_dt = setup.get("magic_candle", {}).get("time")
    if hasattr(magic_candle_dt, "strftime"):
        magic_time_iso = magic_candle_dt.strftime("%Y-%m-%d %H:%M")
        magic_time_str = magic_candle_dt.isoformat()
    else:
        magic_time_iso = str(magic_candle_dt)[:16]
        magic_time_str = str(magic_candle_dt)

    dedup_tag = f"{model_version}:{'suppressed:' if is_suppressed else ''}{symbol}:{direction}:{magic_time_str}"
    dedup_tag_norm = f"{model_version}:{'suppressed:' if is_suppressed else ''}{symbol}:{direction}:{magic_time_iso}"

    # Persistent DB deduplication check
    try:
        from core.database import SignalDatabase
        db = SignalDatabase()
        with db._get_connection() as conn:
            cur = conn.cursor()
            cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
            cur.execute("""
                SELECT id FROM signals 
                WHERE symbol = ? AND signal = ? AND model_version = ?
                  AND timestamp >= ?
                LIMIT 1
            """, (symbol, direction, model_version, cutoff_iso))
            existing_row = cur.fetchone()
            if existing_row:
                return existing_row[0]
    except Exception as dbe:
        logger.warning(f"DB shadow dedup check warning: {dbe}")

    state = _load_confluence_state()
    shadow_setups = state.get("shadow_setups", {})
    if dedup_tag in shadow_setups:
        return shadow_setups[dedup_tag].get("id")
    if dedup_tag_norm in shadow_setups:
        return shadow_setups[dedup_tag_norm].get("id")

    # If this exact candle was placed live on MT5 by this model, skip duplicate shadow record
    placed_setups = state.get("placed_setups", {})
    if f"{model_version}:{symbol}:{direction}:{magic_time_str}" in placed_setups or f"{model_version}:{symbol}:{direction}:{magic_time_iso}" in placed_setups:
        return None

    try:
        from core.database import SignalDatabase
        db = SignalDatabase()
        now_iso = datetime.now(timezone.utc).isoformat()

        conf_val = (float(ml_score) / 100.0) if ml_score is not None else 0.50
        exit_r = "ML_SUPPRESSED" if is_suppressed else "SHADOW_PAPER_TRADE"

        signal_row = {
            "symbol": symbol,
            "signal": direction,
            "expert_signal": direction,
            "confidence": conf_val,
            "confidence_tier": int(round(conf_val * 100)),
            "model_version": model_version,
            "status": "NEW",
            "outcome": "ACTIVE",
            "price_at_signal": setup["entry"],
            "tp_price": setup["tp"],
            "sl_price": setup["sl"],
            "tp_pips": setup.get("tp_pips", 0),
            "sl_pips": setup.get("sl_pips", 0),
            "timestamp": now_iso,
            "mt5_ticket": None,
            "suggested_lots": setup.get("lots", 0.01),
            "order_type": setup.get("order_type", f"{direction}_STOP"),
            "is_hidden": 1,  # 1 = Background shadow trade for performance comparison
            "is_manual": 0,
            "is_proven": 0,
            "regime": "CONFLUENCE",
            "exit_reason": exit_r,
        }
        sig_id = db.save_signal(signal_row)

        shadow_setups[dedup_tag] = {
            "id": sig_id,
            "symbol": symbol,
            "direction": direction,
            "model_version": model_version,
            "suppressed": is_suppressed,
            "recorded_at": now_iso,
        }
        if len(shadow_setups) > 200:
            keys_to_remove = list(shadow_setups.keys())[:-200]
            for k in keys_to_remove:
                shadow_setups.pop(k, None)
        state["shadow_setups"] = shadow_setups
        _save_confluence_state(state)

        mode_label = "🛡️ SUPPRESSED (AI-BLOCKED)" if is_suppressed else "👻 BACKGROUND SHADOW"
        logger.info(f"{mode_label} CONFLUENCE TRADE LOGGED: {symbol} {direction} @ {setup['entry']:.5f} ({model_version}, ID: {sig_id})")
        return sig_id
    except Exception as e:
        logger.warning(f"Could not record confluence shadow trade: {e}")
        return None


def _send_confluence_telegram_alert(setup: Dict[str, Any], submit_res: Dict[str, Any]) -> bool:
    """Send dedicated Telegram alert for automated Confluence entries."""
    try:
        from core.notifications import NotificationManager
        notifier = NotificationManager()
        if not notifier.enabled:
            return False

        sym = setup["symbol"]
        direction = setup["direction"]
        entry = setup["entry"]
        sl = setup["sl"]
        tp = setup["tp"]
        sl_pips = setup["sl_pips"]
        tp_pips = setup["tp_pips"]
        lots = setup["lots"]
        rrr = setup.get("rrr", 1.5)
        ticket = submit_res.get("ticket", "-")
        line_type = setup.get("line_type", "Reference Line")
        reclaimed_line = setup.get("reclaimed_line", 0.0)
        sl_buffer = setup.get("sl_buffer_pips", 2.5)

        arrow = "🟢" if direction == "BUY" else "🔴"
        order_type_str = setup.get("order_type", f"{direction}_STOP")

        broadcast_res = submit_res.get("broadcast_results", {})
        n_accounts = len(broadcast_res) if broadcast_res else 0
        n_success = sum(1 for v in (broadcast_res or {}).values() if str(v).isdigit())
        broadcast_line = f"📡 Broadcast: {n_success}/{n_accounts} accounts executed\n" if n_accounts > 0 else ""

        magic_t = setup["magic_candle"]["time"]
        magic_str = magic_t.strftime("%H:%M UTC") if hasattr(magic_t, "strftime") else str(magic_t)[:16]
        expiry_t = setup["expiry_utc"]
        expiry_str = expiry_t.strftime("%H:%M UTC") if hasattr(expiry_t, "strftime") else "Candle 3 Window"

        ml_info = setup.get("ml_filter", {})
        ml_line = ""
        if ml_info.get("evaluated"):
            ml_score = ml_info.get("confidence_score_pct", 50.0)
            ml_th = ml_info.get("threshold", 0.48) * 100
            ml_line = f"🧠 *ML Quality:* `{ml_score}%` (Gate: `≥{ml_th:.0f}%`)\n"

        model_ver = setup.get("model_version", "confluence_ml_p60")
        from core.dynamic_model_whitelist import get_ytd_model_attribution
        ytd_attr = get_ytd_model_attribution(model_ver, sym)

        if model_ver == "confluence_ml_p60":
            strategy_title = "AUTOMATED CONFLUENCE M15 + AI QUALITY GATE (P60)"
        elif model_ver == "confluence_std_p25":
            strategy_title = "AUTOMATED CONFLUENCE M15 STANDARD (P25)"
        elif model_ver == "confluence_ml_m15":
            strategy_title = "AUTOMATED CONFLUENCE M15 + AI QUALITY GATE"
        else:
            strategy_title = "AUTOMATED CONFLUENCE M15 (RULE-BASED)"

        if ytd_attr["is_ytd"]:
            header_title = f"🏆 *DYNAMIC YTD MODEL SIGNAL*\n⚡ *{strategy_title}*"
            model_block = (
                f"🏆 *Model:* Dynamic YTD Model\n"
                f"⚙️ *Sub-Strategy:* `{ytd_attr['sub_model_name']}`\n"
                f"🛡️ *YTD Gate:* Approved Winning Asset (Net R ≥ 0.0)\n"
            )
        else:
            header_title = f"⚡ *{strategy_title}*"
            model_block = f"📊 *Model:* `{model_ver}`\n"

        be_p = setup.get("be_offset_pips") or get_symbol_be_offset(sym)
        partial_p = setup.get("partial_target_price")
        if partial_p:
            raw_ratio = setup.get("partial_target_ratio") or (0.60 if "ml" in model_ver else 0.25)
            p_ratio = int(round(raw_ratio * 100))
            partial_line = f"💰 *Partial TP:* `{partial_p:.5f}` ({p_ratio}% TP · locks BE+{be_p:.1f}p)\n"
        else:
            partial_line = ""

        msg = (
            f"{header_title}\n"
            f"*{sym}* · `{order_type_str}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{model_block}"
            f"🎯 *Reclamation:* {line_type} (`{reclaimed_line:.5f}`)\n"
            f"🕯️ *Magic Candle:* `{magic_str}`\n"
            f"📍 *Entry:* `{entry:.5f}` (Wick Level)\n"
            f"🛡️ *SL:*    `{sl:.5f}` (`-{sl_pips:.1f}p` · {sl_buffer:.1f}p buffer)\n"
            f"🎯 *TP:*    `{tp:.5f}` (`+{tp_pips:.1f}p`) [1:{rrr} R:R]\n"
            f"{partial_line}"
            f"{ml_line}"
            f"📊 *Volume:* `{lots}` Lots\n"
            f"⏳ *Window:* Until `{expiry_str}` (Candle 3 Expiry)\n"
            f"🎫 *Ticket:* `#{ticket}`\n"
            f"{broadcast_line}"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"_Automated Pending Stop placed on Master & Follower Accounts._"
        )
        return notifier.send_telegram_message(msg)
    except Exception as e:
        logger.warning(f"Confluence alert formatting failed: {e}")
        return False


def _send_partial_profit_telegram_alert(
    symbol: str,
    direction: str,
    ticket: int,
    model_key: str,
    entry_price: float,
    partial_target: float,
    closed_lots: float,
    new_sl: float,
    partial_ratio: float,
    be_offset_pips: float = 2.0,
) -> bool:
    """Send dedicated Telegram alert when partial profit is locked and SL moved to BE+offset."""
    try:
        from core.notifications import NotificationManager
        notifier = NotificationManager()
        if not notifier.enabled:
            return False

        ratio_pct = int(round(partial_ratio * 100))
        arrow = "🟢" if direction == "BUY" else "🔴"
        from core.dynamic_model_whitelist import get_ytd_model_attribution
        ytd_attr = get_ytd_model_attribution(model_key, symbol)

        if ytd_attr["is_ytd"]:
            header_txt = "🏆 *DYNAMIC YTD MODEL · PARTIAL PROFIT SECURED*"
            model_line = (
                f"🏆 *Model:* Dynamic YTD Model\n"
                f"⚙️ *Sub-Strategy:* `{ytd_attr['sub_model_name']}`\n"
            )
        else:
            header_txt = "🎯 *PARTIAL PROFIT SECURED & SL MOVED TO BE*"
            model_name = "🧠 Confluence ML (P60)" if ("p60" in model_key or "ml" in model_key) else "⚡ Confluence Standard (P25)"
            model_line = f"🏷️ *Model:* {model_name}\n"

        msg = (
            f"{header_txt}\n"
            f"{arrow} *{symbol}* · `{direction}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{model_line}"
            f"🎯 *Milestone:* `{ratio_pct}%` of TP Target Hit (`{partial_target:.5f}`)\n"
            f"💰 *Partial Closed:* `{closed_lots}` Lots (50% Volume)\n"
            f"🛡️ *New Stop Loss:* `{new_sl:.5f}` (`+{be_offset_pips:.1f}p` spread buffer locked)\n"
            f"📍 *Entry Price:* `{entry_price:.5f}`\n"
            f"🎫 *Master Ticket:* `#{ticket}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"_Trade is now completely risk-free with spread compensated._"
        )
        return notifier.send_telegram_message(msg)
    except Exception as e:
        logger.warning(f"Partial alert formatting failed: {e}")
        return False


def manage_confluence_open_positions() -> Dict[str, Any]:
    """
    Monitor active Confluence open positions on MT5 Master:
    1. Check if price reached Partial Profit target:
       - confluence_ml_p60: 60% of TP distance
       - confluence_std_p25: 25% of TP distance
    2. When target is reached:
       - Take 50% partial close via MT5 TRADE_ACTION_DEAL
       - Move SL to 2.0 pips away from entry in profit direction (TRADE_ACTION_SLTP) to cover spread
       - Mirror partial close & SL adjustment to all follower accounts via multi_executor
       - Persist state to confluence_state.json
       - Send rich Telegram alert
    """
    results = {"checked": 0, "partials_taken": [], "sl_moved": []}
    mt5 = get_mt5()
    if not mt5:
        return results

    try:
        positions = mt5.positions_get()
        if not positions:
            return results

        state = _load_confluence_state()
        managed_positions = state.setdefault("managed_positions", {})
        dirty = False

        for pos in positions:
            # Check if this position belongs to Confluence models
            comm = str(pos.comment or "")
            magic = getattr(pos, "magic", 0)
            is_confluence = (magic in ALL_APEX_MAGICS) or ("APEX" in comm) or ("CONFLUENCE" in comm)
            if not is_confluence:
                continue

            results["checked"] += 1
            pos_key = str(pos.ticket)
            rec = managed_positions.get(pos_key) or {}

            # Determine model and partial parameters
            # ONLY confluence_ml_p60 and confluence_std_p25 participate in partial profit and BE moves!
            # Original models (confluence_ml_m15 and confluence_m15) run strictly to full TP or SL.
            if "P60" in comm or rec.get("model_version") == "confluence_ml_p60":
                model_key = "confluence_ml_p60"
                partial_ratio = 0.60
            elif "P25" in comm or rec.get("model_version") == "confluence_std_p25":
                model_key = "confluence_std_p25"
                partial_ratio = 0.25
            else:
                # This position belongs to confluence_ml_m15, confluence_m15, or manual/non-partial model.
                # Do NOT take partial profit and do NOT modify SL. Let it run full course to TP or SL.
                continue

            be_offset_pips = rec.get("be_offset_pips") or get_symbol_be_offset(pos.symbol)
            pip_sz = get_pip_size(pos.symbol)
            is_buy = (pos.type == mt5.POSITION_TYPE_BUY)
            direction_str = "BUY" if is_buy else "SELL"
            entry_price = float(pos.price_open)
            tp_price = float(pos.tp) if pos.tp > 0 else float(rec.get("tp", 0.0))

            if tp_price <= 0.0:
                continue

            tp_dist = abs(tp_price - entry_price)
            if tp_dist <= 0:
                continue

            if is_buy:
                partial_target = entry_price + (partial_ratio * tp_dist)
                be_sl = round(entry_price + (be_offset_pips * pip_sz), 5)
                curr_price = float(pos.price_current)
                target_hit = curr_price >= partial_target
            else:
                partial_target = entry_price - (partial_ratio * tp_dist)
                be_sl = round(entry_price - (be_offset_pips * pip_sz), 5)
                curr_price = float(pos.price_current)
                target_hit = curr_price <= partial_target

            partial_taken = rec.get("partial_taken", False)
            be_sl_moved = rec.get("be_sl_moved", False)

            # Broker-level audit: verify if a partial exit deal already executed for this position
            if not partial_taken:
                try:
                    deals = mt5.history_deals_get(position=pos.ticket) or []
                    for d in deals:
                        if getattr(d, "entry", None) in (1, 3) and getattr(d, "profit", 0) != 0:
                            logger.info(f"🛡️ Partial close deal #{d.ticket} already found in MT5 for #{pos.ticket}. Marking partial_taken=True.")
                            partial_taken = True
                            rec["partial_taken"] = True
                            dirty = True
                            break
                except Exception:
                    pass

            # Step 1: Execute partial close if target hit and not already taken
            if target_hit and not partial_taken:
                logger.info(
                    f"🎯 CONFLUENCE PARTIAL TARGET REACHED: {pos.symbol} {direction_str} "
                    f"Current: {curr_price:.5f} >= Target: {partial_target:.5f} ({int(partial_ratio*100)}% of TP for {model_key}). Executing 50% partial close..."
                )
                close_vol = round(pos.volume * 0.5, 2)
                s_info = mt5.symbol_info(pos.symbol)
                vol_min = s_info.volume_min if s_info else 0.01
                vol_step = s_info.volume_step if s_info else 0.01

                partial_ok = False
                closed_lots = 0.0

                if close_vol < vol_min:
                    logger.info(f"Position volume {pos.volume} is at minimum ({vol_min}). Skipping lot close, moving SL directly to BE+2p.")
                    partial_ok = True
                else:
                    close_vol = round(round(close_vol / vol_step) * vol_step, 2)
                    close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
                    tick = mt5.symbol_info_tick(pos.symbol)
                    exec_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                    deal_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": pos.ticket,
                        "symbol": pos.symbol,
                        "volume": close_vol,
                        "type": close_type,
                        "price": exec_price,
                        "deviation": 25,
                        "magic": pos.magic,
                        "comment": f"APEX Partial {int(partial_ratio*100)}%",
                    }
                    deal_res = mt5.order_send(deal_req)
                    if deal_res and deal_res.retcode == mt5.TRADE_RETCODE_DONE:
                        partial_ok = True
                        closed_lots = close_vol
                        logger.info(f"✅ Master partial deal succeeded: Closed {close_vol} lots on #{pos.ticket} ({pos.symbol})")
                    else:
                        err_d = deal_res.comment if deal_res else "No response"
                        logger.error(f"❌ Master partial deal failed for #{pos.ticket}: {err_d}")

                # Step 2: Move SL to BE+2p
                sl_moved_ok = False
                if partial_ok:
                    sltp_req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "symbol": pos.symbol,
                        "sl": be_sl,
                        "tp": pos.tp,
                    }
                    sltp_res = mt5.order_send(sltp_req)
                    if sltp_res and sltp_res.retcode == mt5.TRADE_RETCODE_DONE:
                        sl_moved_ok = True
                        logger.info(f"🛡️ Master SL successfully moved to BE+2p ({be_sl}) for #{pos.ticket} ({pos.symbol})")
                    else:
                        err_s = sltp_res.comment if sltp_res else "No response"
                        logger.warning(f"⚠️ Master SLTP modification returned: {err_s} for #{pos.ticket}")

                    # Step 3: Broadcast partial & SL move to follower accounts
                    try:
                        from scripts.multi_executor import partial_close_and_modify_sl_for_all_users
                        partial_close_and_modify_sl_for_all_users(
                            symbol=pos.symbol,
                            partial_pct=50.0,
                            new_sl=be_sl,
                            model_tag=model_key,
                        )
                    except Exception as me_err:
                        logger.warning(f"Multi-executor follower partial broadcast warning: {me_err}")

                    # Step 4: Record state
                    managed_positions[pos_key] = {
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "direction": direction_str,
                        "model_version": model_key,
                        "entry": entry_price,
                        "tp": tp_price,
                        "partial_target_ratio": partial_ratio,
                        "partial_target_price": partial_target,
                        "closed_lots": closed_lots,
                        "be_sl_price": be_sl,
                        "partial_taken": True,
                        "be_sl_moved": sl_moved_ok,
                        "partial_time": datetime.now(timezone.utc).isoformat(),
                    }
                    dirty = True
                    results["partials_taken"].append({"ticket": pos.ticket, "symbol": pos.symbol, "closed": closed_lots})
                    if sl_moved_ok:
                        results["sl_moved"].append({"ticket": pos.ticket, "new_sl": be_sl})

                    # Step 5: Send Telegram alert
                    _send_partial_profit_telegram_alert(
                        symbol=pos.symbol,
                        direction=direction_str,
                        ticket=pos.ticket,
                        model_key=model_key,
                        entry_price=entry_price,
                        partial_target=partial_target,
                        closed_lots=closed_lots,
                        new_sl=be_sl,
                        partial_ratio=partial_ratio,
                        be_offset_pips=be_offset_pips,
                    )

            # If partial was taken previously but SL was not moved yet, retry moving SL
            elif partial_taken and not be_sl_moved:
                sltp_req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "sl": be_sl,
                    "tp": pos.tp,
                }
                sltp_res = mt5.order_send(sltp_req)
                if sltp_res and sltp_res.retcode == mt5.TRADE_RETCODE_DONE:
                    rec["be_sl_moved"] = True
                    managed_positions[pos_key] = rec
                    dirty = True
                    results["sl_moved"].append({"ticket": pos.ticket, "new_sl": be_sl})
                    logger.info(f"🛡️ Retried and successfully moved SL to BE+2p ({be_sl}) for #{pos.ticket}")

        if dirty:
            state["managed_positions"] = managed_positions
            _save_confluence_state(state)

    except Exception as e:
        logger.error(f"Error in manage_confluence_open_positions: {e}")

    return results


def scan_and_execute_all_pairs(
    enabled_symbols: Optional[List[str]] = None,
    force_auto_trade: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Scan all watchlist pairs for Confluence setups and execute automatically if auto_trade is True.
    """
    cfg = load_confluence_config()
    if not cfg.get("enabled", True):
        return {"scanned": 0, "setups": [], "executed": []}

    # Guardrail Pre-flight Safety Check
    try:
        from core.guardrail import get_guardrail
        guard_res = get_guardrail().get_safety_status()
        if not guard_res.get("safe", True):
            reason = guard_res.get("reason", "Trading blocked by Safety Guardrail.")
            logger.warning(f"⚡ Confluence scanning halted: {reason}")
            return {"scanned": 0, "setups": [], "executed": [], "halted_reason": reason}
    except Exception as e:
        logger.error(f"Guardrail check failed in Confluence scanner: {e}")

    auto_trade = cfg.get("auto_trade", True) if force_auto_trade is None else force_auto_trade
    symbols = enabled_symbols or cfg.get("symbols", DEFAULT_SYMBOLS)

    # Weekend Mode: If Forex/Commodities are halted, filter strictly to crypto symbols (24/7 trading)
    from core.market_hours import is_weekend_halt, is_crypto
    halted, reason = is_weekend_halt()
    if halted:
        symbols = [s for s in symbols if is_crypto(s)]
        if not symbols:
            return {"scanned": 0, "setups": [], "executed": [], "halted_reason": f"Weekend Halt: {reason}"}
        logger.info(f"🪙 Confluence Scanner: Scanning {len(symbols)} crypto pairs 24/7 during weekend ({symbols})")

    from core.model_gatekeeper import is_model_live_authorized
    enable_ml_p60 = is_model_live_authorized("confluence_ml_p60")
    enable_std_p25 = is_model_live_authorized("confluence_std_p25")
    enable_ml_legacy = is_model_live_authorized("confluence_ml_m15")
    enable_std_legacy = is_model_live_authorized("confluence_m15")

    # All known Confluence model variants to evaluate.
    # Models marked live_active place real MT5 pending stop orders.
    # Non-active models run as background SHADOW / PAPER models for performance tracking & comparison.
    models_to_evaluate = [
        ("confluence_ml_p60", bool(enable_ml_p60 and auto_trade)),
        ("confluence_std_p25", bool(enable_std_p25 and auto_trade)),
        ("confluence_ml_m15", bool(enable_ml_legacy and auto_trade)),
        ("confluence_m15", bool(enable_std_legacy and auto_trade)),
    ]

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scanned": len(symbols),
        "setups": [],
        "executed": [],
        "shadow_logged": [],
    }

    for sym in symbols:
        for target_model, is_live_active in models_to_evaluate:
            is_ml_model = ("ml" in target_model.lower())
            ml_enabled = cfg.get("ml_filter_enabled", True) and is_ml_model
            try:
                eval_res = evaluate_confluence_setup(
                    symbol=sym,
                    rrr=cfg.get("rrr", 1.5),
                    sl_buffer_pips=None,  # Automatically adapts per asset class (Gold=25p, Forex=2.5p, etc.)
                    risk_type=cfg.get("risk_type", "percent"),
                    risk_value=cfg.get("risk_value", 0.5),
                    active_model=target_model,
                )
                if eval_res.get("setup_detected"):
                    results["setups"].append(eval_res)
                    ml_info = eval_res.get("ml_filter", {})
                    p_score = ml_info.get("confidence_score_pct", 50.0) if ml_info.get("evaluated") else None
                    score_str = f" [ML: {p_score}%]" if p_score is not None else ""
                    mode_tag = "LIVE" if is_live_active else "SHADOW"
                    logger.info(f"✨ Confluence setup detected for {sym}: {eval_res['direction']} @ {eval_res['entry']}{score_str} ({target_model} · {mode_tag})")

                    # ML Quality Gate Enforcement (Only for ML-based models)
                    if ml_enabled and not ml_info.get("passed", True):
                        th_score = ml_info.get("threshold", 0.48) * 100
                        logger.warning(
                            f"🛡️ ML FILTER BLOCKED TRADE: {sym} {eval_res['direction']} "
                            f"(P(Win) = {p_score}% < {th_score:.0f}% threshold). Logging suppressed shadow trade."
                        )
                        # Always record suppressed trade as shadow trade for later comparison
                        sh_id = record_confluence_shadow_trade(
                            setup=eval_res,
                            model_version=target_model,
                            is_suppressed=True,
                            ml_score=p_score,
                        )
                        if sh_id:
                            results["shadow_logged"].append({"symbol": sym, "id": sh_id, "model": target_model, "suppressed": True})
                        continue

                    # Dynamic Model Whitelist Gate (Winning Assets Only)
                    from core.dynamic_model_whitelist import is_pair_whitelisted_for_model
                    is_pair_winning = is_pair_whitelisted_for_model(target_model, sym)
                    if is_live_active and not is_pair_winning:
                        logger.info(
                            f"🛡️ DYNAMIC YTD GATE: {sym} is not a winning asset for {target_model} (YTD Net R < 0.0). "
                            f"Demoting to background shadow trade."
                        )
                        is_live_active = False

                    # If setup passed (or standard rule-based model):
                    if is_live_active:
                        exec_res = execute_confluence_setup(eval_res)
                        if exec_res.get("success"):
                            results["executed"].append({
                                "symbol": sym,
                                "ticket": exec_res.get("ticket"),
                                "direction": eval_res["direction"],
                                "entry": eval_res["entry"],
                                "model_version": target_model,
                                "ml_score": p_score,
                            })
                    else:
                        # Non-active or benched pair running in the background as SHADOW / PAPER TRADE
                        sh_id = record_confluence_shadow_trade(
                            setup=eval_res,
                            model_version=target_model,
                            is_suppressed=False,
                            ml_score=p_score if is_ml_model else None,
                        )
                        if sh_id:
                            results["shadow_logged"].append({"symbol": sym, "id": sh_id, "model": target_model, "suppressed": False})
            except Exception as e:
                logger.error(f"Error evaluating confluence for {sym} ({target_model}): {e}")

    return results


# Background Scanner Thread Runner
_confluence_watcher_running = False
_confluence_watcher_thread = None


def start_confluence_scanner_watcher():
    """Start background watcher thread that scans pairs every minute and manages open positions continuously."""
    global _confluence_watcher_running, _confluence_watcher_thread
    if _confluence_watcher_running:
        return

    _confluence_watcher_running = True

    def _loop():
        logger.info("⚡ Confluence Automated Scanner Watcher thread started.")
        last_scanned_minute = -1
        while _confluence_watcher_running:
            try:
                # 1. High-frequency position manager: runs every 5 seconds to take partial profit & move SL to BE+2p
                manage_confluence_open_positions()
            except Exception as pe:
                logger.error(f"Error managing confluence positions: {pe}")

            try:
                # 2. Scanner on candle close / every minute
                now_utc = datetime.now(timezone.utc)
                if now_utc.minute != last_scanned_minute:
                    last_scanned_minute = now_utc.minute
                    scan_and_execute_all_pairs()
            except Exception as e:
                logger.error(f"Error in Confluence Watcher loop: {e}")

            time.sleep(5)

    _confluence_watcher_thread = threading.Thread(target=_loop, daemon=True, name="ConfluenceWatcherThread")
    _confluence_watcher_thread.start()
