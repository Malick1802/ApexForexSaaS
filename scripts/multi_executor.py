"""
Multi-User MT5 Executor — executes a signal across ALL registered user accounts.
Called by apex_connect after each new signal is detected.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("MultiExecutor")
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s - MULTI_EXEC - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
logger.setLevel(logging.INFO)


def _get_mt5_module():
    """Return the correct MT5 module depending on OS."""
    import platform
    if platform.system() == "Windows":
        import MetaTrader5 as mt5
        return mt5
    else:
        from mt5linux import MetaTrader5 as mt5
        return mt5()


def _connect_user(mt5, user: dict) -> bool:
    """Initialize MT5 with a specific user's credentials. Returns True if connected."""
    try:
        login = int(user["mt5_login"])
        password = str(user["mt5_password"])
        server = str(user["mt5_server"])
        ok = mt5.initialize(login=login, password=password, server=server)
        if ok:
            acc = mt5.account_info()
            if acc:
                logger.info(f"  ✅ Connected: {user['name']} → {acc.server} (Balance: {acc.balance:.2f})")
                return True
        err = mt5.last_error()
        logger.error(f"  ❌ Failed to connect {user['name']}: {err}")
        return False
    except Exception as e:
        logger.error(f"  ❌ Exception connecting {user['name']}: {e}")
        return False


def _calculate_lots(mt5, user: dict, symbol: str, sl_price: float, entry_price: float) -> float:
    """Calculate lot size based on user's risk settings."""
    risk_type = user.get("risk_type", "fixed")
    risk_value = float(user.get("risk_value", 0.01))

    if risk_type == "fixed":
        return risk_value

    try:
        account_info = mt5.account_info()
        if not account_info:
            return 0.01

        # Percent of balance risk calculation
        risk_amount = account_info.balance * (risk_value / 100.0)
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01

        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        price_dist = abs(entry_price - sl_price)
        dist_in_ticks = price_dist / tick_size if tick_size > 0 else 0

        if dist_in_ticks <= 0 or tick_value <= 0:
            return 0.01

        loss_per_lot = dist_in_ticks * tick_value
        risk_lots = risk_amount / loss_per_lot

        # Clamp to symbol min/max
        step = symbol_info.volume_step
        risk_lots = round(risk_lots / step) * step
        return max(symbol_info.volume_min, min(symbol_info.volume_max, risk_lots))

    except Exception as e:
        logger.error(f"  Lot calculation error: {e}")
        return 0.01


def execute_signal_for_all_users(signal_row: dict) -> dict:
    """
    Execute a single signal across all enabled registered users.
    Returns a summary dict: {user_name: ticket_or_error, ...}
    """
    from core.user_accounts import get_enabled_users, mark_last_trade

    users = get_enabled_users()
    if not users:
        logger.info("No enabled users registered — skipping multi-execution.")
        return {}

    symbol = signal_row["symbol"]
    signal_type = signal_row["signal"]
    sl = float(signal_row.get("sl_price") or 0)
    tp = float(signal_row.get("tp_price") or 0)
    regime = signal_row.get("regime", "NORMAL")

    # ── COMMODITY / BLOCKED SYMBOL SAFETY GATE ──
    from core.symbol_guard import is_symbol_blocked
    if is_symbol_blocked(symbol):
        logger.critical(f"🛑 COMMODITY SHIELD: Symbol {symbol} is a blacklisted commodity. Skipping multi-user execution entirely!")
        return {}

    logger.info(f"🌐 Multi-Executor: Broadcasting {symbol} {signal_type} to {len(users)} account(s)")

    results = {}

    for user in users:
        user_name = user["name"]
        logger.info(f"→ Processing: {user_name}")

        try:
            mt5 = _get_mt5_module()

            if not _connect_user(mt5, user):
                results[user_name] = "CONNECTION_FAILED"
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                continue

            # Select symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"  Symbol {symbol} not visible for {user_name}")
                results[user_name] = "SYMBOL_NOT_FOUND"
                mt5.shutdown()
                continue

            # Get live tick
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"  No tick for {symbol} for {user_name}")
                results[user_name] = "NO_TICK"
                mt5.shutdown()
                continue

            price = tick.ask if signal_type == "BUY" else tick.bid
            order_type = mt5.ORDER_TYPE_BUY if signal_type == "BUY" else mt5.ORDER_TYPE_SELL

            # Calculate lots
            volume = _calculate_lots(mt5, user, symbol, sl, price)

            # Filling mode
            symbol_info = mt5.symbol_info(symbol)
            filling_type = mt5.ORDER_FILLING_FOK
            if symbol_info:
                if (symbol_info.filling_mode & 2) != 0:
                    filling_type = mt5.ORDER_FILLING_IOC

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       volume,
                "type":         order_type,
                "price":        price,
                "sl":           sl,
                "tp":           tp,
                "deviation":    20,
                "magic":        20260622,
                "comment":      f"ForexAlert {regime}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }

            result = mt5.order_send(request)
            mt5.shutdown()

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"  ✅ {user_name}: Ticket {result.order}")
                results[user_name] = result.order
                mark_last_trade(user["id"])
            else:
                comment = result.comment if result else "No response"
                code = result.retcode if result else -1
                logger.error(f"  ❌ {user_name}: {comment} (Code {code})")
                results[user_name] = f"FAILED:{comment}"

        except Exception as e:
            logger.error(f"  ❌ {user_name}: Exception — {e}")
            results[user_name] = f"EXCEPTION:{e}"
            try:
                mt5.shutdown()
            except Exception:
                pass

    logger.info(f"✅ Multi-Executor complete. Results: {results}")

    # ── Telegram Alerts (Option C) ────────────────────────────────────────────
    # Send each subscriber a personal signal alert via Telegram.
    # Works regardless of whether MT5 execution succeeded.
    try:
        from core.telegram_alerts import notify_subscribers
        sent = notify_subscribers(signal_row, execution_results=results)
        if sent:
            logger.info(f"📲 Telegram alerts sent to {sent} subscriber(s)")
    except Exception as _te:
        logger.warning(f"Telegram alert error (non-critical): {_te}")

    return results


def close_signal_for_all_users(symbol: str) -> dict:
    """
    Close all open positions for a symbol across all enabled subscriber accounts.
    Returns a dict mapping username to the status of the close operation.
    """
    from core.user_accounts import get_enabled_users
    users = get_enabled_users()
    if not users:
        return {}

    logger.info(f"🌐 Multi-Executor Reversal: Closing positions for {symbol} across {len(users)} account(s)")
    results = {}

    for user in users:
        user_name = user["name"]
        try:
            mt5 = _get_mt5_module()
            if not _connect_user(mt5, user):
                results[user_name] = "CONNECTION_FAILED"
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                continue

            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                results[user_name] = "NO_OPEN_POSITIONS"
                mt5.shutdown()
                continue

            closed_tickets = []
            for pos in positions:
                ticket = pos.ticket
                pos_type = pos.type
                vol = pos.volume
                
                calc_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                    
                price = tick.bid if calc_type == mt5.ORDER_TYPE_SELL else tick.ask
                
                # Check filling mode
                filling_type = mt5.ORDER_FILLING_IOC
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    if (symbol_info.filling_mode & 2) != 0:
                        filling_type = mt5.ORDER_FILLING_IOC
                    elif (symbol_info.filling_mode & 1) != 0:
                        filling_type = mt5.ORDER_FILLING_FOK

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": vol,
                    "type": calc_type,
                    "position": ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 20260622,
                    "comment": "Apex Reversal Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_type,
                }
                
                res = mt5.order_send(request)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    closed_tickets.append(str(ticket))
                    
            mt5.shutdown()
            if closed_tickets:
                results[user_name] = f"CLOSED:{','.join(closed_tickets)}"
            else:
                results[user_name] = "CLOSE_FAILED"

        except Exception as e:
            logger.error(f"  ❌ {user_name} exception during close: {e}")
            results[user_name] = f"EXCEPTION:{e}"
            try:
                mt5.shutdown()
            except Exception:
                pass

    logger.info(f"✅ Multi-Executor Reversal close complete. Results: {results}")
    return results

