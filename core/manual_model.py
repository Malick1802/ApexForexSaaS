"""
core/manual_model.py
Manual M15 Wick Sniper Model - Discretionary order placement engine.

Execution flow:
1. Fetch last closed 15-minute candle for any symbol via MT5.
2. Compute entry from the wick (High Wick for BUY, Low Wick for SELL).
3. Validate SL orientation.
4. Compute TP from selectable R:R (1:1.5 default or 1:2.0).
5. Calculate broker-accurate lot size from user risk % or fixed $ amount.
6. Place as a Pending Stop Order on the master MT5 account.
7. Broadcast to all enabled copy-trading accounts via execute_signal_for_all_users().
8. Send Telegram alerts: Armed -> Filled -> Resolved.
"""
import logging
import threading
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ManualModel")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_mt5():
    """Return the connected MT5 module for the master account via MT5Connector."""
    try:
        from core.mt5_connector import MT5Connector
        connector = MT5Connector()
        conn = connector.get_connection()
        if conn is not None:
            return conn
        import MetaTrader5 as mt5
        if mt5.terminal_info() is None:
            mt5.initialize()
        return mt5
    except Exception as e:
        logger.error(f"MT5 not available: {e}")
        return None


def get_broker_offset_hours(symbol: str = "EURUSD") -> int:
    """
    Determine the offset in hours between the MT5 broker clock and UTC.
    E.g., FTMO server is GMT+3 (EEST summer), so this returns 3.
    """
    mt5 = get_mt5()
    if not mt5:
        return 0
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick and tick.time > 0:
            now_utc = datetime.now(timezone.utc)
            tick_dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            return round((tick_dt - now_utc).total_seconds() / 3600)
    except Exception:
        pass
    return 0


def get_forming_candle(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Return the CURRENT FORMING M15 candle for the strategy.

    Strategy logic:
    - The user watches the CURRENT forming candle wick as it builds.
    - Entry = HIGH wick (BUY) or LOW wick (SELL) of THIS forming candle.
    - SL    = LOW wick - buffer (BUY) or HIGH wick + buffer (SELL).
    - The order is valid for the NEXT M15 candle only.

    When market is closed (weekend): returns the most recent closed candle
    as a fallback (for UI preview / back-reference purposes).

    Returns dict with:
      open, high, low, close, time (open time UTC), close_time (UTC),
      next_candle_close (when the NEXT M15 window ends - order expiry),
      is_forming (True if candle is still live), spread
    """
    mt5 = get_mt5()
    if not mt5:
        return None
    try:
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Could not select symbol {symbol} in Market Watch.")
            return None

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 5)
        if rates is None or len(rates) < 2:
            logger.error(f"Could not fetch M15 candles for {symbol}: {mt5.last_error()}")
            return None

        now_utc = datetime.now(timezone.utc)
        sym_info = mt5.symbol_info(symbol)
        spread = sym_info.spread * sym_info.point if sym_info else 0.0

        # Calculate broker timezone offset so all displayed times are TRUE UTC
        offset_h = get_broker_offset_hours(symbol)

        # The most recent bar is always rates[-1] (oldest-first array)
        newest = rates[-1]
        raw_server_dt = datetime.fromtimestamp(int(newest["time"]), tz=timezone.utc)
        open_utc  = raw_server_dt - timedelta(hours=offset_h)
        close_utc = open_utc + timedelta(minutes=15)
        is_forming = close_utc > now_utc   # True if candle is still building

        # 5-Minute Grace Window calculation
        elapsed_seconds = max(0.0, (now_utc - open_utc).total_seconds())
        can_arm_previous = (elapsed_seconds <= 300.0)
        grace_seconds_left = max(0, int(300.0 - elapsed_seconds)) if can_arm_previous else 0

        # Previous completed bar is rates[-2]
        prev_bar = rates[-2]
        prev_raw_dt = datetime.fromtimestamp(int(prev_bar["time"]), tz=timezone.utc)
        prev_open_utc = prev_raw_dt - timedelta(hours=offset_h)
        prev_close_utc = prev_open_utc + timedelta(minutes=15)
        prev_open = float(prev_bar["open"])
        prev_high = float(prev_bar["high"])
        prev_low = float(prev_bar["low"])
        prev_close = float(prev_bar["close"])
        prev_is_bullish = prev_close > prev_open
        prev_is_bearish = prev_close < prev_open

        if is_forming:
            # Live market: use live tick for exact current High/Low
            tick = mt5.symbol_info_tick(symbol)
            source = newest
            live_high  = float(source["high"])
            live_low   = float(source["low"])
            live_open  = float(source["open"])
            live_close = tick.bid if tick else float(source["close"])
            next_candle_close = close_utc + timedelta(minutes=15)
        else:
            source = newest
            live_high  = float(source["high"])
            live_low   = float(source["low"])
            live_open  = float(source["open"])
            live_close = float(source["close"])
            next_candle_close = now_utc + timedelta(minutes=15)

        return {
            "symbol":             symbol,
            "open":               live_open,
            "high":               live_high,
            "low":                live_low,
            "close":              live_close,
            "time":               open_utc,
            "close_time":         close_utc,
            "next_candle_close":  next_candle_close,
            "is_forming":         is_forming,
            "spread":             float(spread),
            "elapsed_seconds":    elapsed_seconds,
            "can_arm_previous":   can_arm_previous,
            "grace_seconds_left": grace_seconds_left,
            "previous_candle": {
                "open":       prev_open,
                "high":       prev_high,
                "low":        prev_low,
                "close":      prev_close,
                "open_time":  prev_open_utc,
                "close_time": prev_close_utc,
                "is_bullish": prev_is_bullish,
                "is_bearish": prev_is_bearish,
            },
        }
    except Exception as e:
        logger.error(f"Error fetching forming M15 candle for {symbol}: {e}")
        return None


# Keep the old name as an alias (used by active orders monitor display)
def get_last_15m_candle(symbol: str) -> Optional[Dict[str, Any]]:
    """Alias for get_forming_candle - returns current forming candle."""
    return get_forming_candle(symbol)



def get_pip_size(symbol: str) -> float:
    """Return pip size for a symbol (0.01 for JPY/commodities/SOL, 1.0 for BTC, 0.1 for ETH/Gold, 0.0001 otherwise)."""
    s = symbol.upper().replace(".CASH", "").replace(".M", "").replace(".RAW", "").replace("_", "")
    if "BTC" in s:
        return 1.0
    if "ETH" in s:
        return 0.1
    if "SOL" in s:
        return 0.01
    if any(x in s for x in ("XAU", "GOLD")):
        return 0.1
    if any(x in s for x in ("XAG", "SILVER")):
        return 0.01
    if any(x in s for x in ("OIL", "WTI", "BRENT", "USOIL", "UKOIL")):
        return 0.01
    if "JPY" in s:
        return 0.01
    if any(x in s for x in ("XPT", "XPD", "COPPER")):
        return 0.01
    return 0.0001


def get_asset_max_lots(symbol: str, balance: float) -> float:
    """
    Return maximum safe concurrent volume (lots) allowed across all models for this symbol,
    scaled to account balance, to prevent margin depletion and over-leveraging.
    """
    s = symbol.upper().replace(".CASH", "").replace(".M", "").replace(".RAW", "").replace("_", "")
    scale = max(0.5, (float(balance or 10000.0)) / 10000.0)

    if any(x in s for x in ("OIL", "WTI", "BRENT", "USOIL", "UKOIL")):
        # Oil: 2.0 lots max per $10k balance (~$1,213 margin requirement)
        return max(0.05, round(2.0 * scale, 2))
    elif any(x in s for x in ("XAU", "GOLD")):
        # Gold: 1.5 lots max per $10k balance
        return max(0.02, round(1.5 * scale, 2))
    elif any(x in s for x in ("XAG", "SILVER")):
        # Silver: 2.0 lots max per $10k balance
        return max(0.02, round(2.0 * scale, 2))
    elif any(x in s for x in ("BTC", "ETH", "SOL", "CRYPTO")):
        # Crypto: 2.0 lots max per $10k balance
        return max(0.05, round(2.0 * scale, 2))
    else:
        # Standard Forex pairs: 10.0 lots max per $10k balance
        return max(0.10, round(10.0 * scale, 2))


def calculate_lot_size(
    symbol: str,
    entry: float,
    sl: float,
    risk_type: str = "percent",
    risk_value: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute broker-accurate lot size.
    Returns lots, risk_usd, balance, pip_size, sl_pips, loss_per_lot, margin_per_lot, etc.
    """
    mt5 = get_mt5()
    result = {
        "lots": 0.01, "risk_usd": 0.0, "balance": 0.0,
        "pip_size": get_pip_size(symbol), "sl_pips": 0.0,
        "loss_per_lot": 0.0, "margin_per_lot": 0.0,
        "capped_by_margin": False, "error": None,
    }

    if not mt5:
        result["error"] = "MT5 not connected"
        return result

    try:
        account = mt5.account_info()
        if not account:
            result["error"] = "Cannot read account info"
            return result

        balance = account.balance
        result["balance"] = balance

        r_type = str(risk_type).lower()
        if r_type in ("fixed_cash", "cash", "usd", "fixed_usd", "dollar"):
            risk_usd = float(risk_value)
            result["risk_usd"] = risk_usd
        elif r_type in ("fixed", "fixed_lot", "lot"):
            risk_usd = 0.0  # calculated below once loss_per_lot is known
            result["risk_usd"] = risk_usd
        else:  # "percent"
            risk_usd = balance * (float(risk_value) / 100.0)
            result["risk_usd"] = risk_usd

        sym_info = mt5.symbol_info(symbol)
        if not sym_info:
            result["error"] = f"Symbol {symbol} not found"
            return result

        pip_size = get_pip_size(symbol)
        result["pip_size"] = pip_size
        sl_pips = abs(entry - sl) / pip_size
        result["sl_pips"] = round(sl_pips, 1)

        # Broker-accurate loss per 1.0 lot
        loss_per_lot = None
        try:
            calc_sl = entry - abs(entry - sl)
            profit_1lot = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, entry, calc_sl)
            if profit_1lot is not None and abs(profit_1lot) > 0:
                loss_per_lot = abs(profit_1lot)
        except Exception:
            pass

        if not loss_per_lot or loss_per_lot <= 0:
            tick_size  = sym_info.trade_tick_size or 0.00001
            tick_value = sym_info.trade_tick_value or 1.0
            price_dist = abs(entry - sl)
            loss_per_lot = (price_dist / tick_size) * tick_value

        result["loss_per_lot"] = round(loss_per_lot, 4)

        if r_type in ("fixed", "fixed_lot", "lot"):
            risk_lots = float(risk_value)
            result["risk_usd"] = round(risk_lots * loss_per_lot, 2)
        else:
            risk_lots = risk_usd / loss_per_lot if loss_per_lot > 0 else 0.01

        margin_per_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, entry)
        if not margin_per_lot or margin_per_lot <= 0:
            notional = entry * sym_info.trade_contract_size
            margin_per_lot = notional / 30
        result["margin_per_lot"] = round(margin_per_lot, 2)

        free_margin = max(0.0, getattr(account, "margin_free", balance))
        max_margin_lots = (balance * 0.9) / margin_per_lot if margin_per_lot > 0 else 100.0
        # Dynamic Free Margin Cap: use at most 70% of available free margin to protect open positions
        max_free_margin_lots = (free_margin * 0.70) / margin_per_lot if (free_margin > 0 and margin_per_lot > 0) else 0.0
        capped = (risk_lots > max_margin_lots) or (risk_lots > max_free_margin_lots)
        result["capped_by_margin"] = capped
        raw_lots = min(risk_lots, max_margin_lots, max_free_margin_lots)

        step = sym_info.volume_step or 0.01
        final_lots = round(round(raw_lots / step) * step, 2)
        if final_lots < sym_info.volume_min:
            if max_free_margin_lots < sym_info.volume_min:
                result["error"] = f"Insufficient free margin (${free_margin:.2f}) to open minimum volume {sym_info.volume_min} lots for {symbol} (Requires ${margin_per_lot * sym_info.volume_min:.2f} margin)."
                result["lots"] = 0.0
                return result
            final_lots = sym_info.volume_min
        final_lots = min(sym_info.volume_max, final_lots)
        result["lots"] = final_lots
        return result

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Lot size calculation failed for {symbol}: {e}")
        return result


def calculate_manual_order(
    symbol: str,
    direction: str,           # "BUY" or "SELL"
    rrr: float = 1.5,         # 1.5 or 2.0
    risk_type: str = "percent",
    risk_value: float = 0.5,
    sl_buffer_pips: float = 5.0,   # pips beyond opposite wick (default: 5.0 pips)
    target_candle: str = "forming", # "forming" or "previous"
) -> Dict[str, Any]:
    """
    Build a complete M15 Wick Sniper order specification.

    Strategy:
      - Source: CURRENT FORMING M15 candle OR PREVIOUS CANDLE (within 5-min grace window)
      - BUY:  Entry = candle HIGH wick (BUY STOP)
               SL   = candle LOW wick  - sl_buffer_pips
      - SELL: Entry = candle LOW wick  (SELL STOP)
               SL   = candle HIGH wick + sl_buffer_pips
      - TP:   Entry +/- (SL_distance x RRR)
      - Expiry:
          * If target_candle == 'forming': End of NEXT M15 candle
          * If target_candle == 'previous': End of CURRENT M15 candle

    Color Invalidation Rules:
      - SELL is INVALIDATED if reference candle finishes BULLISH (close > open)
      - BUY is INVALIDATED if reference candle finishes BEARISH (close < open)
    """
    result = {
        "valid": False, "error": None, "warnings": [],
        "symbol": symbol, "direction": direction,
        "entry": None, "sl": None, "tp": None, "rrr": rrr,
        "sl_pips": 0.0, "tp_pips": 0.0, "sl_buffer_pips": sl_buffer_pips,
        "risk_type": risk_type, "risk_value": risk_value,
        "lots": 0.01, "risk_usd": 0.0, "balance": 0.0,
        "candle": None, "order_type": None,
        "expiry_utc": None,
        "target_candle": target_candle,
    }

    # 1. Get current forming candle and previous candle data
    candle = get_forming_candle(symbol)
    if not candle:
        result["error"] = f"Could not fetch M15 data for {symbol}. Check MT5 connection."
        return result
    result["candle"] = candle

    pip_size = get_pip_size(symbol)
    buffer   = sl_buffer_pips * pip_size

    # 2. Entry and SL calculation depending on target candle (Exact wick target)
    if target_candle == "previous":
        if not candle.get("can_arm_previous"):
            result["error"] = (
                f"5-minute grace window for previous candle has expired "
                f"({candle.get('elapsed_seconds', 0):.0f}s elapsed in current candle > 300s limit)."
            )
            return result

        prev = candle.get("previous_candle")
        if not prev:
            result["error"] = "Previous candle data unavailable."
            return result

        # Rule check: Candle Finish Color Invalidation
        if direction == "SELL" and prev.get("is_bullish"):
            result["error"] = (
                f"Previous candle finished BULLISH ({prev['open']:.5f} -> {prev['close']:.5f}). "
                f"SELL orders are prohibited when candle finishes Bullish."
            )
            return result
        if direction == "BUY" and prev.get("is_bearish"):
            result["error"] = (
                f"Previous candle finished BEARISH ({prev['open']:.5f} -> {prev['close']:.5f}). "
                f"BUY orders are prohibited when candle finishes Bearish."
            )
            return result

        if direction == "BUY":
            entry = round(prev["high"], 5)
            sl    = round(prev["low"] - buffer, 5)
            order_type_str = "BUY_STOP"
        else:
            entry = round(prev["low"], 5)
            sl    = round(prev["high"] + buffer, 5)
            order_type_str = "SELL_STOP"

        result["entry"]      = entry
        result["sl"]         = sl
        result["order_type"] = order_type_str
        result["expiry_utc"] = candle["close_time"]   # expires at end of current candle
    else:
        if not candle.get("is_forming"):
            result["warnings"].append(
                "Market is closed - using last available candle as reference. "
                "Order expiry will be set 15 minutes from now."
            )

        if direction == "BUY":
            entry  = round(candle["high"], 5)                  # HIGH wick -> BUY STOP
            sl     = round(candle["low"] - buffer, 5)          # LOW wick - buffer -> SL
            order_type_str = "BUY_STOP"
        else:
            entry  = round(candle["low"], 5)                   # LOW wick -> SELL STOP
            sl     = round(candle["high"] + buffer, 5)         # HIGH wick + buffer -> SL
            order_type_str = "SELL_STOP"

        result["entry"]      = entry
        result["sl"]         = sl
        result["order_type"] = order_type_str
        result["expiry_utc"] = candle["next_candle_close"]

    # 3. SL/TP distances
    sl_dist  = abs(entry - sl)
    sl_pips  = sl_dist / pip_size
    result["sl_pips"] = round(sl_pips, 1)

    if sl_pips < 1.0:
        result["error"] = f"SL distance too small ({sl_pips:.1f} pips). Widen the buffer."
        return result

    tp_dist = sl_dist * rrr
    tp = (entry + tp_dist) if direction == "BUY" else (entry - tp_dist)
    result["tp"]      = round(tp, 5)
    result["tp_pips"] = round(tp_dist / pip_size, 1)

    # 4. Lot sizing
    lot_calc = calculate_lot_size(symbol, entry, sl, risk_type, risk_value)
    result.update({
        "lots":             lot_calc["lots"],
        "risk_usd":         lot_calc["risk_usd"],
        "balance":          lot_calc["balance"],
        "loss_per_lot":     lot_calc.get("loss_per_lot", 0),
        "margin_per_lot":   lot_calc.get("margin_per_lot", 0),
        "capped_by_margin": lot_calc.get("capped_by_margin", False),
    })
    if lot_calc.get("error"):
        result["warnings"].append(f"Lot calc: {lot_calc['error']}")
    if lot_calc.get("capped_by_margin"):
        result["warnings"].append("Lots capped by margin limit.")

    result["projected_gain_usd"] = round(result["risk_usd"] * rrr, 2)
    result["valid"] = True
    return result


def submit_manual_order(
    order_spec: Dict[str, Any],
    broadcast_to_subscribers: bool = True,
    send_telegram: bool = True,
    async_subscribers: bool = True,
) -> Dict[str, Any]:
    """
    Place the manual M15 wick order on MT5 (master account) and optionally
    broadcast to all copy-trading subscriber accounts.
    Order expires at the end of the NEXT M15 candle (ORDER_TIME_SPECIFIED).
    """
    result = {"success": False, "ticket": None, "error": None, "broadcast_results": {}}

    if not order_spec.get("valid"):
        result["error"] = order_spec.get("error", "Invalid order specification.")
        return result

    # Guardrail Safety & Drawdown Check
    try:
        from core.guardrail import get_guardrail
        guard_res = get_guardrail().get_safety_status()
        if not guard_res.get("safe", True):
            reason = guard_res.get("reason", "Trading blocked by Safety Guardrail.")
            logger.warning(f"🛑 Order blocked by Guardrail: {reason}")
            result["error"] = reason
            return result
    except Exception as e:
        logger.warning(f"Error checking guardrail in manual order: {e}")

    # UNIFIED MODEL GATEKEEPER AUTHORIZATION CHECK
    from core.model_gatekeeper import is_model_live_authorized
    mod_ver = order_spec.get("model_version", "manual_m15")
    if not is_model_live_authorized(mod_ver):
        reason = f"Model '{mod_ver}' is currently in SHADOW mode (not selected for live execution)."
        logger.warning(f"🛑 GATEKEEPER BLOCK: {reason}")
        result["error"] = reason
        return result

    # DYNAMIC YTD WINNING ASSET GATE (Only winning assets permitted for live broker submission)
    from core.dynamic_model_whitelist import is_pair_whitelisted_for_model
    sym_to_check = order_spec.get("symbol", "")
    if not is_pair_whitelisted_for_model(mod_ver, sym_to_check):
        reason = f"Symbol '{sym_to_check}' is benched under {mod_ver} Dynamic YTD Whitelist (Net R < 0.0)."
        logger.warning(f"🛑 DYNAMIC YTD GATE BLOCK: {reason}")
        result["error"] = reason
        return result

    mt5 = get_mt5()
    if not mt5:
        result["error"] = "MT5 not connected."
        return result

    # Verify connected account against designated Master Account in database
    from core.user_accounts import get_master_account
    master_rec = get_master_account()
    if master_rec:
        expected_login = str(master_rec.get("mt5_login", "")).strip()
        acc_info = mt5.account_info()
        if acc_info:
            connected_login = str(getattr(acc_info, "login", "")).strip()
            if expected_login and connected_login != expected_login:
                logger.warning(f"⚠️ Connected MT5 account #{connected_login} does not match active Master Account #{expected_login}. Reconnecting...")
                from core.mt5_connector import MT5Connector
                MT5Connector().shutdown()
                mt5 = get_mt5()
                acc_info = mt5.account_info()

            if not master_rec.get("enabled", 1):
                logger.warning(f"🛑 Master Account #{expected_login} is currently paused/disabled. Blocking order placement.")
                result["error"] = f"Master Account #{expected_login} is paused/disabled. Enable it in the Copy Trading Hub to place orders."
                return result

            logger.info(f"🎯 Placing manual order on MT5 Master: #{getattr(acc_info, 'login', '?')} ({getattr(acc_info, 'server', '?')})")

    symbol         = order_spec["symbol"]
    direction      = order_spec["direction"]
    entry          = order_spec["entry"]
    sl             = order_spec["sl"]
    tp             = order_spec["tp"]
    lots           = order_spec["lots"]
    order_type_str = order_spec["order_type"]
    expiry_utc     = order_spec.get("expiry_utc")
    pip_sz         = get_pip_size(symbol)

    mod_ver = str(order_spec.get("model_version") or "manual_m15").lower()
    target_magic = int(order_spec.get("magic") or 202425)
    try:
        from core.confluence_model import MODEL_MAGIC_MAP
        if not order_spec.get("magic"):
            target_magic = MODEL_MAGIC_MAP.get(mod_ver, 202425)
    except Exception:
        pass

    # Master Account Deduplication Guard (Prevents duplicate orders on master MT5 for the SAME model)
    try:
        from core.confluence_model import is_order_from_model
        allow_concurrent = bool(order_spec.get("allow_concurrent_asset", False)) or (mod_ver in ("confluence_ml_p60", "confluence_std_p25"))

        existing_orders = mt5.orders_get(symbol=symbol) or []
        for o in existing_orders:
            comm = str(getattr(o, "comment", "") or "").upper()
            o_magic = getattr(o, "magic", 0)
            is_same_model = is_order_from_model(comm, mod_ver) or (o_magic == target_magic)
            if is_same_model:
                # If entry price is within 3 pips or same direction pending order exists for THIS model, block duplicate
                is_same_dir = (direction == "BUY" and o.type == mt5.ORDER_TYPE_BUY_STOP) or (direction == "SELL" and o.type == mt5.ORDER_TYPE_SELL_STOP)
                if is_same_dir or abs(o.price_open - entry) < (3.0 * pip_sz):
                    logger.warning(f"🛑 MASTER DEDUP: Pending order already active on Master for {symbol} ({direction} @ {o.price_open:.5f}, Ticket #{o.ticket}) by {mod_ver}. Blocking duplicate submission.")
                    result["error"] = f"A pending order is already active on Master at this level for {symbol} by {mod_ver} (Ticket #{o.ticket})."
                    return result
                if not allow_concurrent:
                    result["error"] = f"A pending order is already active on Master for {symbol} by {mod_ver} (Ticket #{o.ticket})."
                    return result

        existing_pos = mt5.positions_get(symbol=symbol) or []
        for p in existing_pos:
            comm = str(getattr(p, "comment", "") or "").upper()
            p_magic = getattr(p, "magic", 0)
            is_same_model = is_order_from_model(comm, mod_ver) or (p_magic == target_magic)
            if is_same_model:
                p_time = getattr(p, "time", 0)
                now_ts = int(datetime.now(timezone.utc).timestamp())
                is_current_candle = (now_ts - p_time) < (15 * 60)
                if not allow_concurrent or is_current_candle:
                    logger.warning(f"🛑 MASTER DEDUP: Position already running on Master for {symbol} {direction} by {mod_ver} (opened {now_ts - p_time}s ago, Ticket #{p.ticket}). Blocking duplicate submission.")
                    result["error"] = f"A position is already open on Master for {symbol} by {mod_ver} (Ticket #{p.ticket})."
                    return result

        now_utc = datetime.now(timezone.utc)
        recent_h_orders = mt5.history_orders_get(now_utc - timedelta(seconds=120), now_utc + timedelta(seconds=10)) or []
        for ho in recent_h_orders:
            comm = str(getattr(ho, "comment", "") or "").upper()
            ho_magic = getattr(ho, "magic", 0)
            is_same_model = is_order_from_model(comm, mod_ver) or (ho_magic == target_magic)
            if ho.symbol == symbol and is_same_model:
                # If placed within 120s at the same level by THIS model, block duplicate
                if abs(ho.price_open - entry) < (3.0 * pip_sz):
                    logger.warning(f"🛑 MASTER DEDUP: Order was recently placed within last 120s for {symbol} by {mod_ver} (Ticket #{ho.ticket}). Blocking duplicate submission.")
                    result["error"] = f"An order was recently placed within 120s for {symbol} by {mod_ver} (Ticket #{ho.ticket})."
                    return result
    except Exception as e:
        logger.warning(f"Master dedup check warning: {e}")

    # ── Combined Asset Exposure Guard (Cross-Model Margin & Exposure Protection) ──
    try:
        from core.confluence_model import ALL_APEX_MAGICS
        account = mt5.account_info()
        acc_bal = getattr(account, "balance", 10000.0) or 10000.0
        max_asset_lots = get_asset_max_lots(symbol, acc_bal)

        current_asset_lots = 0.0
        for p in (mt5.positions_get(symbol=symbol) or []):
            if getattr(p, "magic", 0) in ALL_APEX_MAGICS or "APEX" in str(getattr(p, "comment", "")):
                try:
                    current_asset_lots += float(getattr(p, "volume", 0.0) or 0.0)
                except Exception:
                    pass
        for o in (mt5.orders_get(symbol=symbol) or []):
            if getattr(o, "magic", 0) in ALL_APEX_MAGICS or "APEX" in str(getattr(o, "comment", "")):
                try:
                    current_asset_lots += float(getattr(o, "volume_current", 0.0) or 0.0)
                except Exception:
                    pass

        sym_info = mt5.symbol_info(symbol)
        vol_min = getattr(sym_info, "volume_min", 0.01) or 0.01
        vol_step = getattr(sym_info, "volume_step", 0.01) or 0.01

        if current_asset_lots + lots > max_asset_lots:
            allowed_lots = max_asset_lots - current_asset_lots
            if allowed_lots >= vol_min:
                scaled_lots = round(round(allowed_lots / vol_step) * vol_step, 2)
                if scaled_lots >= vol_min:
                    logger.info(
                        f"⚖️ ASSET EXPOSURE CLAMP for {symbol}: Existing {current_asset_lots:.2f} lots active (Cap: {max_asset_lots:.2f}). "
                        f"Scaling new {mod_ver} order from {lots:.2f} to {scaled_lots:.2f} lots."
                    )
                    lots = scaled_lots
                    order_spec["lots"] = lots
                else:
                    logger.warning(
                        f"🛑 ASSET EXPOSURE CAP REACHED: {symbol} already has {current_asset_lots:.2f}/{max_asset_lots:.2f} lots active. "
                        f"Blocking additional {mod_ver} order to prevent margin exhaustion."
                    )
                    result["error"] = f"Asset exposure cap reached for {symbol} ({current_asset_lots:.2f}/{max_asset_lots:.2f} lots). Order blocked to prevent margin exhaustion."
                    return result
            else:
                logger.warning(
                    f"🛑 ASSET EXPOSURE CAP REACHED: {symbol} already has {current_asset_lots:.2f}/{max_asset_lots:.2f} lots active. "
                    f"Blocking additional {mod_ver} order to prevent margin exhaustion."
                )
                result["error"] = f"Asset exposure cap reached for {symbol} ({current_asset_lots:.2f}/{max_asset_lots:.2f} lots). Order blocked to prevent margin exhaustion."
                return result
    except Exception as exp_err:
        logger.warning(f"Asset exposure check error: {exp_err}")

    tick = mt5.symbol_info_tick(symbol)
    
    # Strictly place pending stop orders. No premature market execution at candle open!
    if direction == "BUY":
        mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
        order_type_str = "BUY_STOP"
        # Ensure entry price is strictly above tick.ask so MT5 accepts BUY_STOP
        if tick and entry <= tick.ask:
            entry = round(tick.ask + 1 * pip_sz, 5)
    else:
        mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
        order_type_str = "SELL_STOP"
        # Ensure entry price is strictly below tick.bid so MT5 accepts SELL_STOP
        if tick and entry >= tick.bid:
            entry = round(tick.bid - 1 * pip_sz, 5)

    filling_type = mt5.ORDER_FILLING_RETURN
    try:
        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            if sym_info.filling_mode & 1:
                filling_type = mt5.ORDER_FILLING_FOK
            elif sym_info.filling_mode & 2:
                filling_type = mt5.ORDER_FILLING_IOC
    except Exception:
        pass

    # Set order to expire at end of Candle 3 (ORDER_TIME_SPECIFIED)
    if expiry_utc:
        type_time  = mt5.ORDER_TIME_SPECIFIED
        offset_h   = get_broker_offset_hours(symbol)
        if isinstance(expiry_utc, str):
            from datetime import datetime as _dt
            try:
                expiry_utc = _dt.fromisoformat(expiry_utc)
            except Exception:
                pass
        server_expiry = expiry_utc + timedelta(hours=offset_h)
        expiration = int(server_expiry.timestamp())
    else:
        type_time  = mt5.ORDER_TIME_DAY
        expiration = 0

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       lots,
        "type":         mt5_order_type,
        "price":        round(entry, 5),
        "sl":           round(sl, 5),
        "tp":           round(tp, 5),
        "magic":        target_magic,
        "comment":      order_spec.get("comment", f"APEX-M15 {direction} {order_spec.get('rrr', 1.5)}R"),
        "type_time":    type_time,
        "type_filling": filling_type,
    }
    if expiration:
        request["expiration"] = expiration

    try:
        mt5_result = mt5.order_send(request)
        if mt5_result and mt5_result.retcode in (10031, 10004):
            logger.warning(f"MT5 returned code {mt5_result.retcode}. Reconnecting MT5Connector and retrying...")
            from core.mt5_connector import MT5Connector
            MT5Connector()._connection = None
            mt5 = get_mt5()
            time.sleep(1)
            mt5_result = mt5.order_send(request)

        if not mt5_result or mt5_result.retcode != mt5.TRADE_RETCODE_DONE:
            err  = mt5_result.comment if mt5_result else "Connection Timeout"
            code = mt5_result.retcode if mt5_result else "N/A"
            logger.error(f"Manual M15 order failed: {err} (Code: {code})")
            result["error"] = f"MT5 Order Failed: {err} (Code: {code})"
            return result

        ticket = mt5_result.order
        result["ticket"]  = ticket
        result["success"] = True
        logger.info(f"ORDER PLACED ({order_spec.get('model_version', 'manual_m15')}): {symbol} {direction} {lots} lots @ {entry} | Ticket: {ticket}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Exception placing order for {symbol}: {e}")
        return result

    # Build signal_row for copier + Telegram
    now_iso = datetime.now(timezone.utc).isoformat()
    signal_row = {
        "id":              None,
        "symbol":          symbol,
        "signal":          direction,
        "confidence":      1.0,
        "price_at_signal": entry,
        "sl_price":        sl,
        "tp_price":        tp,
        "sl_pips":         order_spec.get("sl_pips", 0),
        "tp_pips":         order_spec.get("tp_pips", 0),
        "timestamp":       now_iso,
        "mt5_ticket":      ticket,
        "model_version":   order_spec.get("model_version", "manual_m15"),
        "suggested_lots":  lots,
        "risk_value":      order_spec.get("risk_value", 0.5),
        "risk_type":       order_spec.get("risk_type", "percent"),
        "projected_gain":  order_spec.get("projected_gain_usd", 0.0),
        "rrr":             order_spec.get("rrr", 1.5),
        "order_type":      order_type_str,
        "is_hidden":       0,
        "is_manual":       1,
        "is_proven":       1,
        "regime":          order_spec.get("regime", "CONFLUENCE" if order_spec.get("model_version") == "confluence_m15" else "MANUAL"),
        "candle_time":     order_spec.get("candle", {}).get("time", now_iso),
        "expiry_utc":      order_spec.get("expiry_utc"),
        "sl_buffer_pips":  order_spec.get("sl_buffer_pips", 5.0),
    }

    # Save to DB
    try:
        from core.database import SignalDatabase
        db = SignalDatabase()
        sig_id = db.save_signal(signal_row)
        signal_row["id"] = sig_id
    except Exception as e:
        logger.warning(f"Could not save manual signal to DB: {e}")

    # Broadcast to subscriber copy-trading accounts (Decoupled background execution)
    broadcast_results = {}
    if broadcast_to_subscribers:
        try:
            account = mt5.account_info() if mt5 else None
            m_balance = float(getattr(account, "balance", 0.0) or 0.0) if account else 0.0
            signal_row["master_balance"] = m_balance
            signal_row["master_volume"]  = lots

            def _bg_subscriber_broadcast(sig_dict, do_telegram):
                try:
                    import scripts.multi_executor
                    b_res = scripts.multi_executor.execute_signal_for_all_users(sig_dict)
                    logger.info(f"🌐 Background follower copy broadcast finished for {sig_dict.get('symbol')} ({sig_dict.get('model_version')}): {b_res}")
                    if do_telegram:
                        try:
                            _send_manual_telegram_alert(sig_dict, b_res)
                        except Exception as te:
                            logger.warning(f"Telegram alert error in background broadcaster: {te}")
                except Exception as bge:
                    logger.error(f"Background subscriber broadcast error: {bge}")

            if async_subscribers:
                t = threading.Thread(
                    target=_bg_subscriber_broadcast,
                    args=(dict(signal_row), send_telegram),
                    daemon=True,
                    name=f"SubscriberBroadcast-{symbol}-{ticket}"
                )
                t.start()
                broadcast_results = {"status": "DISPATCHED_BACKGROUND"}
                result["broadcast_results"] = broadcast_results
            else:
                import scripts.multi_executor
                broadcast_results = scripts.multi_executor.execute_signal_for_all_users(signal_row)
                result["broadcast_results"] = broadcast_results
                if send_telegram:
                    try:
                        _send_manual_telegram_alert(signal_row, broadcast_results)
                    except Exception as e:
                        logger.warning(f"Telegram alert for manual M15 order failed: {e}")
        except Exception as e:
            logger.error(f"Manual M15 subscriber broadcast setup error: {e}")
    elif send_telegram:
        try:
            _send_manual_telegram_alert(signal_row, broadcast_results)
        except Exception as e:
            logger.warning(f"Telegram alert for manual M15 order failed: {e}")

    return result


def _send_manual_telegram_alert(signal_row: dict, broadcast_results: dict = None) -> bool:
    """Send a richly formatted Telegram alert for a manual M15 wick order."""
    try:
        from core.notifications import NotificationManager
        notifier = NotificationManager()
        if not notifier.enabled:
            return False

        sym        = signal_row.get("symbol", "?")
        direction  = signal_row.get("signal", "?")
        entry      = signal_row.get("price_at_signal", 0)
        sl         = signal_row.get("sl_price", 0)
        tp         = signal_row.get("tp_price", 0)
        sl_pips    = signal_row.get("sl_pips", 0)
        tp_pips    = signal_row.get("tp_pips", 0)
        lots       = signal_row.get("suggested_lots", 0)
        risk_val   = signal_row.get("risk_value", 0.5)
        risk_type  = signal_row.get("risk_type", "percent")
        rrr        = signal_row.get("rrr", 1.5)
        ticket     = signal_row.get("mt5_ticket", "-")
        gain       = signal_row.get("projected_gain", 0)
        order_type = signal_row.get("order_type", "PENDING")
        candle_time = signal_row.get("candle_time", "")
        expiry_utc = signal_row.get("expiry_utc")
        sl_buffer  = signal_row.get("sl_buffer_pips", 5.0)

        arrow      = "🟢" if direction == "BUY" else "🔴"
        if risk_type == "percent":
            risk_label = f"{float(risk_val):.2f}%"
        elif str(risk_type).lower() in ("fixed_cash", "cash", "usd", "fixed_usd", "dollar"):
            risk_label = f"${float(risk_val):.2f}"
        else:
            risk_label = f"{float(risk_val):.2f} lots"

        n_accounts = len(broadcast_results) if broadcast_results else 0
        n_success  = sum(1 for v in (broadcast_results or {}).values() if str(v).isdigit())
        broadcast_line = f"📡 Broadcast: {n_success}/{n_accounts} accounts executed\n" if n_accounts > 0 else ""

        try:
            from datetime import timedelta as _td
            if hasattr(candle_time, "strftime"):
                candle_ts = (candle_time + _td(minutes=15)).strftime("%H:%M UTC")
            else:
                candle_ts = str(candle_time)[:16]
        except Exception:
            candle_ts = str(candle_time)[:16]

        mod_ver    = signal_row.get("model_version", "manual_m15")
        from core.dynamic_model_whitelist import get_ytd_model_attribution
        ytd_attr   = get_ytd_model_attribution(mod_ver, sym)

        if ytd_attr["is_ytd"]:
            header_txt = f"🏆 *DYNAMIC YTD MODEL · MANUAL M15 WICK ORDER ARMED*"
            model_line = (
                f"🏆 *Model:* Dynamic YTD Model\n"
                f"⚙️ *Sub-Strategy:* `{ytd_attr['sub_model_name']}`\n"
                f"🛡️ *YTD Gate:* Approved Winning Asset (Net R ≥ 0.0)\n"
            )
        else:
            header_txt = f"{arrow} *MANUAL M15 WICK ORDER ARMED*"
            model_line = f"📊 *Model:* `{mod_ver}`\n"

        msg = (
            f"{header_txt}\n"
            f"*{sym}* · `{order_type}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{model_line}"
            f"Forming Candle Wick: `{candle_ts}`\n"
            f"Entry:  `{entry:.5f}` (Wick Level)\n"
            f"SL:     `{sl:.5f}` (`-{sl_pips:.1f}p` · {sl_buffer:.1f}p buffer)\n"
            f"TP:     `{tp:.5f}` (`+{tp_pips:.1f}p`) [1:{rrr}]\n"
            f"Risk:   `{risk_label}` · Lots: `{lots}`\n"
            f"Gain:   `+${gain:.2f}`\n"
            f"Expiry: `{expiry_str}` (Next M15 Close)\n"
            f"Ticket: `#{ticket}`\n"
            f"{broadcast_line}"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"_Triggers if next candle reaches the wick._"
        )
        return notifier.send_telegram_message(msg)
    except Exception as e:
        logger.warning(f"Manual Telegram alert failed: {e}")
        return False


def get_active_manual_orders() -> List[Dict[str, Any]]:
    """Return all active manual M15 pending + open positions across Apex models."""
    mt5 = get_mt5()
    if not mt5:
        return []
    orders = []

    apex_magics = (202425, 202404, 202460, 202415, 202401)
    try:
        from core.confluence_model import ALL_APEX_MAGICS
        apex_magics = ALL_APEX_MAGICS
    except Exception:
        pass

    try:
        pending = mt5.orders_get()
        if pending:
            for o in pending:
                if o.magic in apex_magics or "APEX" in (o.comment or ""):
                    orders.append({
                        "type":          "PENDING",
                        "ticket":        o.ticket,
                        "symbol":        o.symbol,
                        "direction":     "BUY" if o.type in (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_LIMIT) else "SELL",
                        "order_type_str": {
                            mt5.ORDER_TYPE_BUY_STOP:   "BUY STOP",
                            mt5.ORDER_TYPE_SELL_STOP:  "SELL STOP",
                            mt5.ORDER_TYPE_BUY_LIMIT:  "BUY LIMIT",
                            mt5.ORDER_TYPE_SELL_LIMIT: "SELL LIMIT",
                        }.get(o.type, "PENDING"),
                        "entry":     o.price_open,
                        "sl":        o.sl,
                        "tp":        o.tp,
                        "lots":      o.volume_current,
                        "placed_at": datetime.fromtimestamp(o.time_setup, tz=timezone.utc).isoformat(),
                        "comment":   o.comment,
                        "pnl":       None,
                    })
    except Exception as e:
        logger.warning(f"Could not fetch pending M15 orders: {e}")

    try:
        positions = mt5.positions_get()
        if positions:
            for p in positions:
                if p.magic in apex_magics or "APEX" in (p.comment or ""):
                    orders.append({
                        "type":          "OPEN",
                        "ticket":        p.ticket,
                        "symbol":        p.symbol,
                        "direction":     "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                        "order_type_str": "LIVE",
                        "entry":         p.price_open,
                        "sl":            p.sl,
                        "tp":            p.tp,
                        "lots":          p.volume,
                        "pnl":           p.profit,
                        "placed_at":     datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                        "comment":       p.comment,
                    })
    except Exception as e:
        logger.warning(f"Could not fetch open M15 positions: {e}")

    return orders


def cancel_manual_order(ticket: int) -> Dict[str, Any]:
    """Cancel a pending manual M15 order by ticket number."""
    mt5 = get_mt5()
    if not mt5:
        return {"success": False, "error": "MT5 not connected"}
    try:
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Manual M15 order #{ticket} cancelled.")
            try:
                from core.database import SignalDatabase
                db = SignalDatabase()
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM signals WHERE mt5_ticket = ?", (str(ticket),))
                    row = cursor.fetchone()
                    if row:
                        db.update_signal_outcome(row[0], 'CANCELLED', exit_reason='User Cancelled')
            except Exception as dbe:
                logger.warning(f"Could not update DB for cancelled ticket {ticket}: {dbe}")
            return {"success": True, "ticket": ticket}
        return {"success": False, "error": res.comment if res else "Unknown error"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def close_manual_position(ticket: int) -> Dict[str, Any]:
    """Emergency close an active open position by ticket number."""
    mt5 = get_mt5()
    if not mt5:
        return {"success": False, "error": "MT5 not connected"}
    try:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Position #{ticket} not found"}
        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": 202425,
            "comment": "M15 Manual Close",
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            outcome = 'SUCCESS' if pos.profit >= 0 else 'FAIL'
            try:
                from core.database import SignalDatabase
                db = SignalDatabase()
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM signals WHERE mt5_ticket = ?", (str(ticket),))
                    row = cursor.fetchone()
                    if row:
                        db.update_signal_outcome(row[0], outcome, exit_price=price, exit_reason=f"Closed Manually (${pos.profit:.2f})")
            except Exception as dbe:
                logger.warning(f"Could not update DB for closed position {ticket}: {dbe}")
            return {"success": True, "ticket": ticket}
        return {"success": False, "error": res.comment if res else "Order close failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Armed M15 Sniper Queue & Auto-Execution Watcher
# =============================================================================
import msvcrt

class ProcessLock:
    """
    Cross-process non-blocking / timed file lock for Windows using msvcrt.
    Guarantees strictly single-instance execution across separate Python processes.
    """
    def __init__(self, lock_file: Path, timeout: float = 0.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self.file_handle = None

    def acquire(self) -> bool:
        start_t = time.time()
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.file_handle = open(self.lock_file, "a+")
                self.file_handle.seek(0)
                msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except (OSError, IOError, PermissionError):
                if self.file_handle:
                    try:
                        self.file_handle.close()
                    except Exception:
                        pass
                    self.file_handle = None
                if time.time() - start_t >= self.timeout:
                    return False
                time.sleep(0.05)

    def release(self):
        if self.file_handle:
            try:
                self.file_handle.seek(0)
                msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None

    def __enter__(self):
        if not self.acquire():
            raise BlockingIOError(f"Could not acquire process lock on {self.lock_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


ARMED_QUEUE_PATH  = PROJECT_ROOT / "data" / "armed_m15_orders.json"
WATCHER_LOCK_PATH = PROJECT_ROOT / "data" / "armed_watcher.lock"
QUEUE_LOCK_PATH   = PROJECT_ROOT / "data" / "armed_queue.lock"
_watcher_thread   = None
_watcher_lock     = threading.Lock()


def _load_armed_queue() -> List[Dict[str, Any]]:
    if not ARMED_QUEUE_PATH.exists():
        return []
    try:
        with open(ARMED_QUEUE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading armed orders queue: {e}")
        return []


def _save_armed_queue(queue: List[Dict[str, Any]]):
    try:
        ARMED_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARMED_QUEUE_PATH, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving armed orders queue: {e}")


def get_armed_orders() -> List[Dict[str, Any]]:
    """Return all active armed M15 sniper tasks waiting for candle close."""
    queue = _load_armed_queue()
    return [item for item in queue if item.get("status") == "ARMED_WAITING_CLOSE"]


def disarm_order(arm_id: str) -> Dict[str, Any]:
    """Cancel an armed setup before the candle completes."""
    queue_lock = ProcessLock(QUEUE_LOCK_PATH, timeout=5.0)
    with queue_lock:
        queue = _load_armed_queue()
        found = False
        for item in queue:
            if item.get("arm_id") == arm_id and item.get("status") in ("ARMED_WAITING_CLOSE", "PROCESSING"):
                item["status"] = "CANCELLED"
                item["cancelled_at"] = datetime.now(timezone.utc).isoformat()
                found = True
                break
        if found:
            _save_armed_queue(queue)
            logger.info(f"M15 sniper order {arm_id} disarmed.")
            return {"success": True, "arm_id": arm_id}
        return {"success": False, "error": f"Armed task {arm_id} not found."}


def arm_m15_order(
    symbol: str,
    direction: str,
    rrr: float = 1.5,
    risk_type: str = "percent",
    risk_value: float = 0.5,
    sl_buffer_pips: float = 5.0,
    broadcast_to_subscribers: bool = True,
    send_telegram: bool = True,
    target_candle: str = "forming",  # "forming" or "previous"
) -> Dict[str, Any]:
    """
    Pre-arm an M15 Wick Sniper setup.
    - If target_candle == 'previous': executes immediately on the previous candle's wicks if within the 5-minute grace window.
    - If target_candle == 'forming': pre-arms while the candle is forming, locking wicks and validating candle finish color upon close.
    Guarded by cross-process ProcessLock and in-place deduplication to prevent duplicate submissions.
    """
    candle = get_forming_candle(symbol)
    if not candle:
        return {"success": False, "error": f"Cannot fetch M15 candle for {symbol}."}

    now_utc = datetime.now(timezone.utc)

    # ── Branch 1: 5-Minute Grace Window on Previous Candle ──
    if target_candle == "previous":
        if not candle.get("can_arm_previous"):
            return {
                "success": False,
                "error": f"5-minute grace window has expired ({candle.get('elapsed_seconds', 0):.0f}s elapsed in current candle > 300s limit)."
            }

        order_spec = calculate_manual_order(
            symbol=symbol,
            direction=direction,
            rrr=rrr,
            risk_type=risk_type,
            risk_value=risk_value,
            sl_buffer_pips=sl_buffer_pips,
            target_candle="previous",
        )
        if not order_spec.get("valid"):
            return {"success": False, "error": order_spec.get("error", "Invalid order spec.")}

        # Place immediately on MT5 & copy accounts
        res = submit_manual_order(
            order_spec=order_spec,
            broadcast_to_subscribers=broadcast_to_subscribers,
            send_telegram=send_telegram
        )
        if not res.get("success"):
            return {"success": False, "error": res.get("error", "Order placement failed.")}

        arm_id = f"ARM_PREV_{symbol}_{int(now_utc.timestamp())}"
        task = {
            "arm_id": arm_id,
            "symbol": symbol,
            "direction": direction,
            "rrr": rrr,
            "risk_type": risk_type,
            "risk_value": risk_value,
            "sl_buffer_pips": sl_buffer_pips,
            "target_candle": "previous",
            "target_close_time": candle["close_time"].isoformat(),
            "broadcast_to_subscribers": broadcast_to_subscribers,
            "send_telegram": send_telegram,
            "status": "PLACED",
            "created_at": now_utc.isoformat(),
            "placed_at": now_utc.isoformat(),
            "preview_entry": order_spec["entry"],
            "entry": order_spec["entry"],
            "sl": order_spec["sl"],
            "tp": order_spec["tp"],
            "lots": order_spec["lots"],
            "ticket": res.get("ticket"),
            "expiry_utc": candle["close_time"].isoformat(),
            "error": None,
        }

        queue_lock = ProcessLock(QUEUE_LOCK_PATH, timeout=5.0)
        with queue_lock:
            queue = _load_armed_queue()
            queue.append(task)
            _save_armed_queue(queue)

        logger.info(f"🎯 M15 SNIPER PLACED ON PREVIOUS CANDLE (5-MIN GRACE): {symbol} {direction} | Ticket #{res.get('ticket')}")

        return {
            "success": True,
            "placed_immediately": True,
            "arm_id": arm_id,
            "ticket": res.get("ticket"),
            "target_candle": "previous",
            "target_close_time": candle["close_time"].isoformat(),
            "target_close_str": candle["close_time"].strftime("%H:%M UTC"),
        }

    # ── Branch 2: Standard Pre-Arm on Current Forming Candle ──
    target_close = candle.get("close_time", now_utc + timedelta(minutes=15))

    queue_lock = ProcessLock(QUEUE_LOCK_PATH, timeout=5.0)
    with queue_lock:
        queue = _load_armed_queue()
        cutoff = now_utc - timedelta(hours=24)
        cleaned_queue = []
        existing_armed = None

        for item in queue:
            try:
                created = datetime.fromisoformat(item["created_at"])
                if created > cutoff or item.get("status") in ("ARMED_WAITING_CLOSE", "PROCESSING"):
                    # Check for existing armed setup on the same symbol
                    if item.get("symbol") == symbol and item.get("status") in ("ARMED_WAITING_CLOSE", "PROCESSING"):
                        existing_armed = item
                    cleaned_queue.append(item)
            except Exception:
                pass

        if existing_armed:
            logger.info(f"⚠️ M15 SNIPER DEDUP: {symbol} is already armed ({existing_armed['arm_id']}). Updating in-place.")
            existing_armed["direction"] = direction
            existing_armed["rrr"] = rrr
            existing_armed["risk_type"] = risk_type
            existing_armed["risk_value"] = risk_value
            existing_armed["sl_buffer_pips"] = sl_buffer_pips
            existing_armed["target_candle"] = "forming"
            existing_armed["target_close_time"] = target_close.isoformat()
            existing_armed["broadcast_to_subscribers"] = broadcast_to_subscribers
            existing_armed["send_telegram"] = send_telegram
            existing_armed["preview_entry"] = candle["high"] if direction == "BUY" else candle["low"]
            _save_armed_queue(cleaned_queue)
            return {
                "success": True,
                "already_armed": True,
                "arm_id": existing_armed["arm_id"],
                "target_candle": "forming",
                "target_close_time": target_close.isoformat(),
                "target_close_str": target_close.strftime("%H:%M UTC"),
            }

        arm_id = f"ARM_{symbol}_{int(now_utc.timestamp())}"
        task = {
            "arm_id": arm_id,
            "symbol": symbol,
            "direction": direction,
            "rrr": rrr,
            "risk_type": risk_type,
            "risk_value": risk_value,
            "sl_buffer_pips": sl_buffer_pips,
            "target_candle": "forming",
            "target_close_time": target_close.isoformat(),
            "broadcast_to_subscribers": broadcast_to_subscribers,
            "send_telegram": send_telegram,
            "status": "ARMED_WAITING_CLOSE",
            "created_at": now_utc.isoformat(),
            "preview_entry": candle["high"] if direction == "BUY" else candle["low"],
            "ticket": None,
            "error": None,
        }

        cleaned_queue.append(task)
        _save_armed_queue(cleaned_queue)

    logger.info(f"🎯 M15 SNIPER ARMED: {symbol} {direction} | Target close: {target_close.strftime('%H:%M:%S UTC')}")

    if send_telegram:
        try:
            from core.notifications import NotificationManager
            notifier = NotificationManager()
            if notifier.enabled:
                arrow = "🟢" if direction == "BUY" else "🔴"
                cancel_rule_str = "Bullish" if direction == "SELL" else "Bearish"
                msg = (
                    f"🎯 *M15 WICK SNIPER PRE-ARMED*\n"
                    f"*{symbol}* · `{direction}` (Targeting Next M15 Candle)\n"
                    f"Target Candle Close: `{target_close.strftime('%H:%M UTC')}`\n"
                    f"R:R Ratio: `1:{rrr}` · Risk: `{risk_value:.2f}%`\n"
                    f"🛡️ *Auto-Cancel Rule*: If candle finishes *{cancel_rule_str}*, order will be automatically cancelled.\n"
                    f"If valid, system captures locked wicks and submits `{direction}_STOP` pending order."
                )
                notifier.send_telegram_message(msg)
        except Exception as e:
            logger.warning(f"Pre-arm Telegram notice failed: {e}")

    start_armed_sniper_watcher()

    return {
        "success": True,
        "placed_immediately": False,
        "arm_id": arm_id,
        "target_candle": "forming",
        "target_close_time": target_close.isoformat(),
        "target_close_str": target_close.strftime("%H:%M UTC"),
    }


def process_armed_m15_orders() -> List[Dict[str, Any]]:
    """
    Check all armed M15 setups. When a candle's close time is reached:
    Capture the locked wicks and submit the pending stop order on MT5.
    Guarded by cross-process ProcessLock to ensure strictly single-instance atomic execution.
    """
    queue_lock = ProcessLock(QUEUE_LOCK_PATH, timeout=0.0)
    if not queue_lock.acquire():
        return []

    try:
        queue = _load_armed_queue()
        now_utc = datetime.now(timezone.utc)

        # Clean up stale "PROCESSING" tasks older than 3 minutes
        for task in queue:
            if task.get("status") == "PROCESSING":
                proc_time_str = task.get("processing_started_at")
                if proc_time_str:
                    try:
                        proc_t = datetime.fromisoformat(proc_time_str)
                        if (now_utc - proc_t).total_seconds() > 180:
                            task["status"] = "EXPIRED"
                            task["error"] = "Processing timeout (exceeded 3 minutes)"
                            logger.warning(f"Reset stale PROCESSING task {task.get('arm_id')} to EXPIRED.")
                    except Exception:
                        task["status"] = "EXPIRED"

        armed_items = [i for i in queue if i.get("status") == "ARMED_WAITING_CLOSE"]
        if not armed_items:
            _save_armed_queue(queue)
            return []

        processed_results = []

        for task in armed_items:
            try:
                target_close = datetime.fromisoformat(task["target_close_time"])
            except Exception:
                continue

            if now_utc >= target_close:
                symbol = task["symbol"]
                direction = task["direction"]

                # Guardrail Safety & Drawdown Check
                try:
                    from core.guardrail import get_guardrail
                    guard_res = get_guardrail().get_safety_status()
                    if not guard_res.get("safe", True):
                        reason = guard_res.get("reason", "Trading blocked by Safety Guardrail.")
                        logger.warning(f"🛑 Armed setup placement blocked by Guardrail: {reason}")
                        task["status"] = "BLOCKED"
                        task["error"] = reason
                        _save_armed_queue(queue)
                        processed_results.append(task)
                        continue
                except Exception as e:
                    logger.error(f"Guardrail check failed in process_armed_m15_orders: {e}")

                mt5 = get_mt5()
                if not mt5:
                    task["status"] = "FAILED"
                    task["error"] = "MT5 not connected"
                    _save_armed_queue(queue)
                    continue

                # Broker-Level Pre-flight Deduplication:
                # Check if an order or position or recent fill ALREADY exists on MT5 for this symbol
                existing_orders = mt5.orders_get(symbol=symbol) or []
                existing_pos = mt5.positions_get(symbol=symbol) or []
                recent_orders = []
                try:
                    recent_orders = mt5.history_orders_get(
                        now_utc - timedelta(seconds=90),
                        now_utc + timedelta(seconds=10)
                    ) or []
                except Exception:
                    pass

                task_mod = str(task.get("model_version") or "manual_m15").lower()
                try:
                    from core.confluence_model import is_order_from_model, MODEL_MAGIC_MAP
                    task_magic = int(task.get("magic") or MODEL_MAGIC_MAP.get(task_mod, 202401))
                except Exception:
                    task_magic = 202401
                    is_order_from_model = lambda c, m: ("APEX-M15" in str(c or "").upper())

                existing_ticket = None
                for o in existing_orders:
                    o_comm = str(getattr(o, "comment", "") or "").upper()
                    if is_order_from_model(o_comm, task_mod) or (getattr(o, "magic", 0) == task_magic):
                        existing_ticket = o.ticket
                        break
                if not existing_ticket:
                    for p in existing_pos:
                        p_comm = str(getattr(p, "comment", "") or "").upper()
                        if is_order_from_model(p_comm, task_mod) or (getattr(p, "magic", 0) == task_magic):
                            existing_ticket = p.ticket
                            break
                if not existing_ticket:
                    for ho in recent_orders:
                        ho_comm = str(getattr(ho, "comment", "") or "").upper()
                        if ho.symbol == symbol and (is_order_from_model(ho_comm, task_mod) or getattr(ho, "magic", 0) == task_magic):
                            existing_ticket = ho.ticket
                            break

                if existing_ticket:
                    logger.warning(f"🛑 DEDUP: Active MT5 order/position #{existing_ticket} already exists for {symbol}. Marking armed task as PLACED.")
                    task["status"] = "PLACED"
                    task["ticket"] = existing_ticket
                    task["placed_at"] = now_utc.isoformat()
                    _save_armed_queue(queue)
                    processed_results.append(task)
                    continue

                # ATOMIC MUTEX: Instantly mark task as PROCESSING and save to disk
                # BEFORE placing order so no other cycle or thread can pick it up!
                task["status"] = "PROCESSING"
                task["processing_started_at"] = now_utc.isoformat()
                _save_armed_queue(queue)

                rrr = task["rrr"]
                risk_type = task["risk_type"]
                risk_value = task["risk_value"]
                sl_buffer_pips = task.get("sl_buffer_pips", 5.0)
                broadcast = task.get("broadcast_to_subscribers", True)
                send_tg = task.get("send_telegram", True)

                logger.info(f"⏰ CANDLE CLOSED for armed {task['arm_id']}: Finalizing wicks and submitting pending order...")

                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 5)
                if rates is None or len(rates) < 2:
                    task["status"] = "FAILED"
                    task["error"] = "Could not fetch closed candle from MT5"
                    _save_armed_queue(queue)
                    continue

                offset_h = get_broker_offset_hours(symbol)
                closed_candle = None
                closed_close_dt = target_close
                for r in reversed(rates):
                    raw_dt = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
                    open_dt = raw_dt - timedelta(hours=offset_h)
                    close_dt = open_dt + timedelta(minutes=15)
                    if close_dt <= now_utc:
                        closed_candle = r
                        closed_close_dt = close_dt
                        break

                if not closed_candle:
                    task["status"] = "FAILED"
                    task["error"] = "No finalized closed candle found"
                    _save_armed_queue(queue)
                    continue

                # ── Candle Finish Color Invalidation Rule ──
                # If SELL and candle closed BULLISH (close > open): cancel order!
                # If BUY and candle closed BEARISH (close < open): cancel order!
                c_open = float(closed_candle["open"])
                c_close = float(closed_candle["close"])
                is_bullish = c_close > c_open
                is_bearish = c_close < c_open

                if direction == "SELL" and is_bullish:
                    reason = f"Candle finished BULLISH ({c_open:.5f} -> {c_close:.5f}). SELL order invalidated & cancelled."
                    logger.info(f"🚫 ARMED SETUP CANCELLED: {symbol} SELL - {reason}")
                    task["status"] = "CANCELLED"
                    task["cancelled_at"] = now_utc.isoformat()
                    task["error"] = reason
                    _save_armed_queue(queue)
                    processed_results.append(task)

                    if send_tg:
                        try:
                            from core.notifications import NotificationManager
                            notifier = NotificationManager()
                            if notifier.enabled:
                                msg = (
                                    f"🚫 *M15 WICK SNIPER CANCELLED*\n"
                                    f"*{symbol}* · `SELL` Invalidated\n"
                                    f"Candle closed *BULLISH* (`{c_open:.5f}` ➔ `{c_close:.5f}`).\n"
                                    f"Rule: SELL order cancelled automatically and NOT placed on MT5."
                                )
                                notifier.send_telegram_message(msg)
                        except Exception as e:
                            logger.warning(f"Cancellation Telegram notice failed: {e}")
                    continue

                if direction == "BUY" and is_bearish:
                    reason = f"Candle finished BEARISH ({c_open:.5f} -> {c_close:.5f}). BUY order invalidated & cancelled."
                    logger.info(f"🚫 ARMED SETUP CANCELLED: {symbol} BUY - {reason}")
                    task["status"] = "CANCELLED"
                    task["cancelled_at"] = now_utc.isoformat()
                    task["error"] = reason
                    _save_armed_queue(queue)
                    processed_results.append(task)

                    if send_tg:
                        try:
                            from core.notifications import NotificationManager
                            notifier = NotificationManager()
                            if notifier.enabled:
                                msg = (
                                    f"🚫 *M15 WICK SNIPER CANCELLED*\n"
                                    f"*{symbol}* · `BUY` Invalidated\n"
                                    f"Candle closed *BEARISH* (`{c_open:.5f}` ➔ `{c_close:.5f}`).\n"
                                    f"Rule: BUY order cancelled automatically and NOT placed on MT5."
                                )
                                notifier.send_telegram_message(msg)
                        except Exception as e:
                            logger.warning(f"Cancellation Telegram notice failed: {e}")
                    continue

                high_wick = float(closed_candle["high"])
                low_wick = float(closed_candle["low"])
                pip_sz = get_pip_size(symbol)
                buffer = sl_buffer_pips * pip_sz

                if direction == "BUY":
                    entry = round(high_wick, 5)
                    sl = round(low_wick - buffer, 5)
                    order_type_str = "BUY_STOP"
                else:
                    entry = round(low_wick, 5)
                    sl = round(high_wick + buffer, 5)
                    order_type_str = "SELL_STOP"

                sl_dist = abs(entry - sl)
                tp_dist = sl_dist * rrr
                tp = round((entry + tp_dist) if direction == "BUY" else (entry - tp_dist), 5)
                next_close = closed_close_dt + timedelta(minutes=15)

                lot_calc = calculate_lot_size(symbol, entry, sl, risk_type, risk_value)

                order_spec = {
                    "valid": True,
                    "error": None,
                    "symbol": symbol,
                    "direction": direction,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "rrr": rrr,
                    "sl_pips": round(sl_dist / pip_sz, 1),
                    "tp_pips": round(tp_dist / pip_sz, 1),
                    "sl_buffer_pips": sl_buffer_pips,
                    "risk_type": risk_type,
                    "risk_value": risk_value,
                    "lots": lot_calc["lots"],
                    "risk_usd": lot_calc["risk_usd"],
                    "balance": lot_calc["balance"],
                    "order_type": order_type_str,
                    "expiry_utc": next_close,
                    "candle": {
                        "time": datetime.fromtimestamp(int(closed_candle["time"]), tz=timezone.utc),
                        "high": high_wick,
                        "low": low_wick,
                        "close": float(closed_candle["close"]),
                    }
                }

                res = submit_manual_order(
                    order_spec=order_spec,
                    broadcast_to_subscribers=broadcast,
                    send_telegram=send_tg
                )

                if res.get("success"):
                    task["status"] = "PLACED"
                    task["ticket"] = res.get("ticket")
                    task["placed_at"] = now_utc.isoformat()
                    task["entry"] = entry
                    task["sl"] = sl
                    task["tp"] = tp
                    task["lots"] = order_spec["lots"]
                    task["expiry_utc"] = next_close.isoformat()
                    logger.info(f"✅ ARMED ORDER PLACED on MT5: Ticket #{res.get('ticket')} | {symbol} {direction} @ {entry:.5f}")
                else:
                    task["status"] = "FAILED"
                    task["error"] = res.get("error", "Order placement failed")
                    logger.error(f"❌ Failed to place armed order {task['arm_id']}: {task['error']}")

                _save_armed_queue(queue)
                processed_results.append(task)

        return processed_results
    finally:
        queue_lock.release()


def reconcile_manual_orders():
    """
    Check all active manual orders in the database against MT5.
    If a pending order filled, track position.
    If a position or order was closed (TP hit, SL hit, or expired),
    query history and update SignalDatabase outcome to SUCCESS / FAIL / CANCELLED.
    """
    try:
        from core.database import SignalDatabase
        db = SignalDatabase()
        with db._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals WHERE (is_manual = 1 OR model_version IN ('manual_m15', 'confluence_m15')) AND outcome IN ('ACTIVE', 'N/A')")
            active_manual_signals = [dict(r) for r in cursor.fetchall()]
        
        if not active_manual_signals:
            return

        mt5 = get_mt5()
        if not mt5:
            return

        for sig in active_manual_signals:
            ticket_str = sig.get('mt5_ticket')
            sig_id = sig['id']
            symbol = sig.get('symbol')
            if not ticket_str or not str(ticket_str).isdigit():
                continue
            ticket = int(ticket_str)

            # 1. Check if still pending in MT5
            pending_orders = mt5.orders_get(ticket=ticket)
            if pending_orders:
                # Active Candle 3 Expiry Check: If Candle 3 window has ended, actively remove pending order
                expiry_val = sig.get('expiry_utc')
                if expiry_val:
                    try:
                        if isinstance(expiry_val, str):
                            from datetime import datetime as _dt
                            exp_dt = _dt.fromisoformat(expiry_val)
                        else:
                            exp_dt = expiry_val
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                        now_utc = datetime.now(timezone.utc)
                        if now_utc >= exp_dt:
                            logger.info(f"⏳ Pending order #{ticket} for {symbol} reached Candle 3 expiry ({exp_dt}). Removing order from MT5.")
                            cancel_manual_order(ticket)
                            db.update_signal_outcome(sig_id, 'EXPIRED', exit_reason='Candle 3 Expired (Untouched)')
                            continue
                    except Exception as _ex:
                        logger.warning(f"Candle 3 expiry check error for ticket #{ticket}: {_ex}")
                continue  # Still within Candle 3 window, waiting to trigger

            # 2. Check if open position in MT5
            open_pos = mt5.positions_get(ticket=ticket)
            if open_pos:
                continue  # Still open position

            # Also check if positions_get(symbol=symbol) has a position belonging to this trade/model
            open_sym_pos = mt5.positions_get(symbol=symbol)
            if open_sym_pos:
                sig_mod = str(sig.get('model_version') or 'manual_m15').lower()
                try:
                    from core.confluence_model import is_order_from_model, MODEL_MAGIC_MAP
                    sig_magic = MODEL_MAGIC_MAP.get(sig_mod, 202425)
                except Exception:
                    sig_magic = 202425
                    is_order_from_model = lambda c, m: True
                has_active = any(
                    getattr(p, 'ticket', 0) == ticket or 
                    (getattr(p, 'magic', 0) == sig_magic and is_order_from_model(getattr(p, 'comment', ''), sig_mod))
                    for p in open_sym_pos
                )
                if has_active:
                    continue

            # 3. If neither pending order nor open position exists, it was resolved!
            # Query history deals first (if filled position closed)
            hist_deals = mt5.history_deals_get(position=ticket)
            if hist_deals:
                exit_deals = [d for d in hist_deals if getattr(d, 'entry', None) in (1, 2, 3)]
                if not exit_deals:
                    # Only entry deal exists — position is still actively open on broker, do NOT resolve
                    continue
                total_profit = sum(getattr(d, 'profit', 0.0) for d in hist_deals)
                exit_price = exit_deals[-1].price
                outcome = 'SUCCESS' if total_profit > 0 else 'FAIL'
                reason = f"M15 TP hit (+${total_profit:.2f})" if total_profit > 0 else f"M15 SL hit (${total_profit:.2f})"
                db.update_signal_outcome(sig_id, outcome, exit_price=exit_price, exit_reason=reason)
                logger.info(f"Reconciled manual trade #{ticket} (ID {sig_id}): {outcome} ({reason})")
                continue

            # If no deals, check order history (cancelled or expired pending order)
            hist_orders = mt5.history_orders_get(ticket=ticket)
            if hist_orders:
                o_state = hist_orders[0].state
                if o_state == mt5.ORDER_STATE_CANCELED:
                    db.update_signal_outcome(sig_id, 'CANCELLED', exit_reason='Order Cancelled')
                    logger.info(f"Reconciled manual order #{ticket} (ID {sig_id}): CANCELLED")
                elif o_state == mt5.ORDER_STATE_EXPIRED:
                    db.update_signal_outcome(sig_id, 'EXPIRED', exit_reason='Candle Expired (Unfilled)')
                    logger.info(f"Reconciled manual order #{ticket} (ID {sig_id}): EXPIRED")
                else:
                    db.update_signal_outcome(sig_id, 'CANCELLED', exit_reason=f'Order State {o_state}')
            else:
                db.update_signal_outcome(sig_id, 'EXPIRED', exit_reason='Untracked order flush')
    except Exception as e:
        logger.error(f"Error reconciling manual orders: {e}")


def _watcher_loop():
    """Background thread loop that checks armed orders every 2 seconds and reconciles manual trades."""
    watcher_lock = ProcessLock(WATCHER_LOCK_PATH, timeout=0.0)
    if not watcher_lock.acquire():
        logger.info("ℹ️ Another process/thread holds the Armed Sniper Watcher lock. Watcher thread exiting cleanly.")
        return

    logger.info("M15 Armed Sniper background watcher thread started with exclusive OS process lock.")
    try:
        while True:
            # Real-time Prop Firm Drawdown Kill Switch check (every 2s)
            try:
                from core.guardrail import get_guardrail
                get_guardrail().get_safety_status()
            except Exception as e:
                logger.error(f"Error checking guardrail in watcher loop: {e}")

            try:
                process_armed_m15_orders()
            except Exception as e:
                logger.error(f"Error in M15 sniper watcher loop: {e}")
            try:
                reconcile_manual_orders()
            except Exception as e:
                logger.error(f"Error in manual orders reconciliation: {e}")
            time.sleep(2)
    finally:
        watcher_lock.release()


def start_armed_sniper_watcher():
    """Ensure the background watcher thread is running (single thread per process, single process across OS)."""
    global _watcher_thread
    with _watcher_lock:
        if _watcher_thread is None or not _watcher_thread.is_alive():
            _watcher_thread = threading.Thread(target=_watcher_loop, daemon=True, name="M15SniperWatcher")
            _watcher_thread.start()


