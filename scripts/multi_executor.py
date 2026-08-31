"""
Multi-User MT5 Executor — executes a signal across ALL registered user accounts.
Uses isolated subprocess workers to guarantee zero IPC interference with the master FTMO scanner.
"""
import logging
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("MultiExecutor")
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s - MULTI_EXEC - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
logger.setLevel(logging.INFO)


def _get_terminal_path_for_server(server: str) -> str:
    """Return the correct terminal64.exe path based on broker server."""
    if "FTMO" in server.upper():
        ftmo_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
        if Path(ftmo_path).exists():
            return ftmo_path
    std_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if Path(std_path).exists():
        return std_path
    return ""


def _worker_execute_order(user: dict, signal_row: dict) -> dict:
    """
    Executed inside an ISOLATED worker subprocess.
    Maintains a private MT5 context that never touches the master process.
    """
    import MetaTrader5 as mt5

    login = int(user["mt5_login"])
    password = str(user["mt5_password"])
    server = str(user["mt5_server"])
    user_name = user.get("name", f"User_{login}")

    symbol = signal_row["symbol"]
    signal_type = signal_row["signal"].upper()
    sl = float(signal_row.get("sl_price") or signal_row.get("stop_loss") or signal_row.get("sl") or 0)
    tp = float(signal_row.get("tp_price") or signal_row.get("take_profit") or signal_row.get("tp") or 0)
    regime = signal_row.get("regime", "NORMAL")

    term_path = _get_terminal_path_for_server(server)

    logger.info(f"Worker init: {user_name} (#{login}) on {server} via terminal [{term_path}]")

    try:
        mt5.shutdown()
    except Exception:
        pass

    init_kwargs = {"timeout": 5000}
    if term_path:
        init_kwargs["path"] = term_path

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        logger.error(f"❌ Worker MT5 initialize failed for {user_name}: {err}")
        return {"status": "FAILED", "error": f"INIT_FAILED_{err}"}

    acc = mt5.account_info()
    if not acc or acc.login != login:
        logger.info(f"Switching login to #{login} on {server}...")
        if not mt5.login(login=login, password=password, server=server):
            err = mt5.last_error()
            logger.error(f"❌ Worker login failed for #{login} on {server}: {err}")
            mt5.shutdown()
            return {"status": "FAILED", "error": f"LOGIN_FAILED_{err}"}
        acc = mt5.account_info()

    if not acc or acc.login != login:
        logger.error(f"❌ Account mismatch in worker! Target #{login}, but connected to #{getattr(acc, 'login', 'None')}")
        mt5.shutdown()
        return {"status": "FAILED", "error": "ACCOUNT_MISMATCH"}

    logger.info(f"✅ Worker logged in: {acc.name} (#{acc.login}) on {acc.server} | Balance: ${acc.balance:,.2f}")

    # Check if position already open on this account (Deduplication)
    existing_pos = mt5.positions_get(symbol=symbol)
    if existing_pos:
        for p in existing_pos:
            if (signal_type == "BUY" and p.type == 0) or (signal_type == "SELL" and p.type == 1):
                logger.warning(f"🛑 DEDUP: {symbol} {signal_type} already open on #{login} (Ticket #{p.ticket}). Blocking duplicate.")
                mt5.shutdown()
                return {"status": "SKIPPED", "ticket": p.ticket, "reason": "ALREADY_OPEN"}

    if not mt5.symbol_select(symbol, True):
        logger.error(f"❌ Symbol {symbol} not available on {server}")
        mt5.shutdown()
        return {"status": "FAILED", "error": "SYMBOL_NOT_FOUND"}

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"❌ No live tick for {symbol}")
        mt5.shutdown()
        return {"status": "FAILED", "error": "NO_TICK"}

    price = tick.ask if signal_type == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal_type == "BUY" else mt5.ORDER_TYPE_SELL

    # Dynamic 0.5% risk lot size calculation
    risk_value = float(user.get("risk_value", 0.5))
    risk_amount = acc.balance * (risk_value / 100.0)

    s_info = mt5.symbol_info(symbol)
    default_pips = 0.28 if "JPY" in symbol else 0.0028
    if sl <= 0:
        price_dist = default_pips
    else:
        price_dist = abs(price - sl)
        if price_dist > (price * 0.05):
            price_dist = default_pips

    tick_size = s_info.trade_tick_size or 0.00001
    tick_val = s_info.trade_tick_value or 1.0
    dist_in_ticks = price_dist / tick_size if tick_size > 0 else 0
    loss_per_lot = dist_in_ticks * tick_val if (dist_in_ticks > 0 and tick_val > 0) else 1.0

    raw_lots = risk_amount / loss_per_lot
    step = s_info.volume_step or 0.01
    volume = round(raw_lots / step) * step
    volume = max(s_info.volume_min, min(s_info.volume_max, volume))

    # Supported filling mode
    filling_type = mt5.ORDER_FILLING_FOK
    if s_info:
        if (s_info.filling_mode & 1) != 0:
            filling_type = mt5.ORDER_FILLING_FOK
        elif (s_info.filling_mode & 2) != 0:
            filling_type = mt5.ORDER_FILLING_IOC
        else:
            filling_type = mt5.ORDER_FILLING_RETURN

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       volume,
        "type":         order_type,
        "price":        price,
        "sl":           sl if sl > 0 else 0.0,
        "tp":           tp if tp > 0 else 0.0,
        "deviation":    30,
        "magic":        20260622,
        "comment":      f"ForexAlert {regime}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    logger.info(f"📤 Placing order: {symbol} {signal_type} | Lots: {volume:.2f} (Risk: ${risk_amount:.2f}) | Filling: {filling_type}")
    res = mt5.order_send(request)

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"  ✅ SUCCESS: Placed {symbol} {signal_type} {volume:.2f} lots! Ticket #{res.order}")
        mt5.shutdown()
        return {"status": "SUCCESS", "ticket": res.order, "volume": volume}
    else:
        comment = res.comment if res else "No response"
        code = res.retcode if res else -1
        logger.error(f"  ❌ FAILED: {comment} (Code: {code})")
        mt5.shutdown()
        return {"status": "FAILED", "error": f"{comment} ({code})"}


def _worker_close_order(user: dict, symbol: str) -> dict:
    """Executed inside an ISOLATED worker subprocess to close positions for a symbol."""
    import MetaTrader5 as mt5

    login = int(user["mt5_login"])
    password = str(user["mt5_password"])
    server = str(user["mt5_server"])
    user_name = user.get("name", f"User_{login}")

    term_path = _get_terminal_path_for_server(server)

    try:
        mt5.shutdown()
    except Exception:
        pass

    init_kwargs = {"login": login, "password": password, "server": server, "timeout": 15000}
    if term_path:
        init_kwargs["path"] = term_path

    if not mt5.initialize(**init_kwargs):
        return {"status": "FAILED", "error": "INIT_FAILED"}

    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        mt5.shutdown()
        return {"status": "NO_OPEN_POSITIONS"}

    closed_tickets = []
    for pos in positions:
        ticket = pos.ticket
        vol = pos.volume
        calc_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        price = tick.bid if calc_type == mt5.ORDER_TYPE_SELL else tick.ask

        s_info = mt5.symbol_info(symbol)
        filling_type = mt5.ORDER_FILLING_IOC
        if s_info:
            if (s_info.filling_mode & 2) != 0:
                filling_type = mt5.ORDER_FILLING_IOC
            elif (s_info.filling_mode & 1) != 0:
                filling_type = mt5.ORDER_FILLING_FOK
            else:
                filling_type = mt5.ORDER_FILLING_RETURN

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": vol,
            "type": calc_type,
            "position": ticket,
            "price": price,
            "deviation": 30,
            "magic": 20260622,
            "comment": "ForexAlert Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            closed_tickets.append(str(ticket))

    mt5.shutdown()
    return {"status": "CLOSED", "tickets": closed_tickets}


def execute_signal_for_all_users(signal_row: dict) -> dict:
    """
    Broadcast a signal across all enabled user accounts using ISOLATED worker subprocesses.
    The calling process (main.py / executive.py) NEVER disconnects from FTMO.
    """
    from core.user_accounts import get_enabled_users, mark_last_trade

    users = get_enabled_users()
    if not users:
        logger.info("No enabled copy trading users registered.")
        return {}

    symbol = signal_row["symbol"]
    signal_type = signal_row["signal"]

    from core.symbol_guard import is_symbol_blocked
    if is_symbol_blocked(symbol):
        logger.critical(f"🛑 COMMODITY SHIELD: Symbol {symbol} is blacklisted. Skipping multi-user execution!")
        return {}

    logger.info(f"🌐 Multi-Executor (Subprocess Isolated): Broadcasting {symbol} {signal_type} to {len(users)} user(s)")

    results = {}

    for user in users:
        user_name = user["name"]
        user_id = user["id"]
        logger.info(f"→ Spawning isolated worker for: {user_name} (#{user['mt5_login']})")

        cmd = [
            sys.executable,
            "-X", "utf8",
            "-m", "scripts.multi_executor",
            "--worker-exec",
            "--user-json", json.dumps(dict(user)),
            "--signal-json", json.dumps(dict(signal_row))
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(PROJECT_ROOT))
            out = proc.stdout.strip()
            if proc.returncode == 0:
                logger.info(f"  Worker Output for {user_name}:\n{out}")
                # Parse result json if in output
                results[user_name] = "SUCCESS"
                mark_last_trade(user_id)
            else:
                logger.error(f"  Worker error for {user_name} (Exit code {proc.returncode}):\n{proc.stderr.strip()}\n{out}")
                results[user_name] = f"ERROR: {proc.stderr.strip()}"
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ Worker timed out for {user_name}")
            results[user_name] = "TIMEOUT"
        except Exception as _we:
            logger.error(f"  ❌ Worker exception for {user_name}: {_we}")
            results[user_name] = f"EXCEPTION: {_we}"

    logger.info(f"✅ Multi-Executor broadcast finished. Results: {results}")

    # Send personal Telegram alerts to subscribers
    try:
        from core.telegram_alerts import notify_subscribers
        notify_subscribers(signal_row, execution_results=results)
    except Exception as _te:
        logger.warning(f"Telegram alert error: {_te}")

    return results


def close_signal_for_all_users(symbol: str) -> dict:
    """Close positions for a symbol across all enabled users via isolated subprocesses."""
    from core.user_accounts import get_enabled_users

    users = get_enabled_users()
    if not users:
        return {}

    logger.info(f"🌐 Multi-Executor (Subprocess Isolated): Closing {symbol} for {len(users)} user(s)")
    results = {}

    for user in users:
        user_name = user["name"]
        cmd = [
            sys.executable,
            "-X", "utf8",
            "-m", "scripts.multi_executor",
            "--worker-close",
            "--user-json", json.dumps(dict(user)),
            "--symbol", symbol
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(PROJECT_ROOT))
            if proc.returncode == 0:
                results[user_name] = "CLOSED"
            else:
                results[user_name] = f"ERROR: {proc.stderr.strip()}"
        except Exception as _ce:
            results[user_name] = f"EXCEPTION: {_ce}"

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-exec", action="store_true", help="Run order execution worker")
    parser.add_argument("--worker-close", action="store_true", help="Run close worker")
    parser.add_argument("--user-json", type=str, help="User credentials JSON string")
    parser.add_argument("--signal-json", type=str, help="Signal details JSON string")
    parser.add_argument("--symbol", type=str, help="Symbol to close")
    args = parser.parse_args()

    if args.worker_exec and args.user_json and args.signal_json:
        user_data = json.loads(args.user_json)
        sig_data = json.loads(args.signal_json)
        res = _worker_execute_order(user_data, sig_data)
        print(json.dumps(res))
        sys.exit(0 if res.get("status") in ("SUCCESS", "SKIPPED") else 1)

    elif args.worker_close and args.user_json and args.symbol:
        user_data = json.loads(args.user_json)
        res = _worker_close_order(user_data, args.symbol)
        print(json.dumps(res))
        sys.exit(0 if res.get("status") in ("CLOSED", "NO_OPEN_POSITIONS") else 1)
