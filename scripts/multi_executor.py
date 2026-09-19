"""
Multi-User MT5 Executor — executes a signal across ALL registered user accounts.
Uses isolated subprocess workers to guarantee zero IPC interference with the master FTMO scanner.
"""
import logging
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8 for console output on Windows to prevent Emoji crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("MultiExecutor")
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s - MULTI_EXEC - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(str(PROJECT_ROOT / "executive.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter('%(asctime)s - MULTI_EXEC - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    except Exception:
        pass
logger.setLevel(logging.INFO)


def _get_terminal_path_for_server(server: str, user_terminal_path: str = "") -> str:
    """Return the correct terminal64.exe path based on user preference, broker server, or installed MT5."""
    # 1. Custom / personalized terminal path explicitly configured by user
    if user_terminal_path and Path(user_terminal_path).exists():
        return str(Path(user_terminal_path))

    srv = (server or "").upper()

    # 2. FTMO detection
    if "FTMO" in srv:
        ftmo_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
        if Path(ftmo_path).exists():
            return ftmo_path

    # 3. Check for common broker branded terminal folders
    common_broker_patterns = [
        (r"C:\Program Files\IC Markets MT5\terminal64.exe", ["ICMARKET"]),
        (r"C:\Program Files\Pepperstone MT5\terminal64.exe", ["PEPPERSTONE"]),
        (r"C:\Program Files\Eightcap MT5\terminal64.exe", ["EIGHTCAP"]),
        (r"C:\Program Files\FundedNext MT5\terminal64.exe", ["FUNDEDNEXT"]),
        (r"C:\Program Files\TopTier MT5\terminal64.exe", ["TOPTIER"]),
        (r"C:\Program Files\Vantage MT5\terminal64.exe", ["VANTAGE"]),
        (r"C:\Program Files\XM MT5\terminal64.exe", ["XM"]),
    ]
    for p_str, kws in common_broker_patterns:
        if any(k in srv for k in kws) and Path(p_str).exists():
            return p_str

    # 4. Standard MetaTrader 5 fallback
    std_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if Path(std_path).exists():
        return std_path

    # 5. Program Files (x86) fallback
    x86_path = r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
    if Path(x86_path).exists():
        return x86_path

    return ""


def resolve_symbol_for_server(symbol: str, mt5_conn) -> str:
    """
    Resolve broker-specific symbol aliases and suffixes across different brokers.
    E.g. USOIL.cash -> USOIL / XTIUSD / WTI / CRUDEOIL / USOIL.raw / USOILm,
    while strictly rejecting non-commodity equities (e.g. WTI stock shares).
    """
    if not symbol:
        return symbol

    def _is_invalid_equity(cand_info, orig_sym: str) -> bool:
        if not cand_info:
            return True
        path_lower = getattr(cand_info, 'path', '').lower()
        desc_lower = getattr(cand_info, 'description', '').lower()
        # If original symbol is a commodity/forex, reject equity stocks/ETFs/trusts
        if any(c in orig_sym.upper() for c in ["OIL", "BRENT", "WTI", "XAU", "GOLD", "XAG", "SILVER"]):
            bad_terms = ["stock", "etf", "shares", "nasdaq", "nyse", "equit", " inc", " corp", " ltd", " plc", "trust", "fund", "adr"]
            if any(b in path_lower or b in desc_lower for b in bad_terms):
                if not any(k in path_lower for k in ["commodit", "energi", "energy", "metals", "futures", "cfd"]):
                    return True
                if any(b in desc_lower for b in [" inc", " corp", " ltd", " plc", "trust", "shares"]):
                    return True
        return False

    clean = symbol.upper().strip()

    # 1. Direct match
    s_info = mt5_conn.symbol_info(symbol)
    if s_info and not _is_invalid_equity(s_info, clean):
        return symbol

    # Common alias groups (roots) across global brokers
    OIL_WTI_ROOTS = ["USOIL", "XTIUSD", "WTI", "CRUDEOIL", "CL", "WTISPOT", "USOUSD", "OIL_CRUDE", "OIL"]
    OIL_BRENT_ROOTS = ["UKOIL", "XBRUSD", "BRENT", "BRENTOIL", "BRENTSPOT", "UKOUSD", "BRN"]
    GOLD_ROOTS = ["XAUUSD", "GOLD", "XAUEUR", "XAUAUD"]
    SILVER_ROOTS = ["XAGUSD", "SILVER"]

    candidate_roots = []
    if any(k in clean for k in ["USOIL", "WTI", "CRUDE", "CL", "XTI"]):
        candidate_roots = OIL_WTI_ROOTS
    elif any(k in clean for k in ["UKOIL", "BRENT", "BRN", "XBR"]):
        candidate_roots = OIL_BRENT_ROOTS
    elif any(k in clean for k in ["XAU", "GOLD"]):
        candidate_roots = GOLD_ROOTS
    elif any(k in clean for k in ["XAG", "SILVER"]):
        candidate_roots = SILVER_ROOTS
    else:
        # Forex pairs: strip .cash or custom suffixes
        base_clean = clean.split(".")[0].split("_")[0]
        candidate_roots = [clean, base_clean]

    # 2. Try exact alias roots
    for cand in candidate_roots:
        cand_info = mt5_conn.symbol_info(cand)
        if cand_info and not _is_invalid_equity(cand_info, clean):
            return cand

    # 3. Match all broker symbols against candidate roots (handling suffixes/prefixes: .raw, .pro, m, +, etc.)
    all_syms = mt5_conn.symbols_get()
    if all_syms:
        # Priority 3a: Symbol starts with or matches candidate root (e.g. XTIUSD.raw, USOILm, WTI.cash)
        for cand in candidate_roots:
            for s in all_syms:
                s_name = s.name.upper()
                if not _is_invalid_equity(s, clean):
                    if s_name == cand or s_name.startswith(cand) or s_name.endswith(cand):
                        return s.name

        # Priority 3b: Semantic description match for Oil/Commodities (strictly within commodity/energy categories)
        if candidate_roots in (OIL_WTI_ROOTS, OIL_BRENT_ROOTS):
            target_term = "BRENT" if candidate_roots == OIL_BRENT_ROOTS else "CRUDE"
            for s in all_syms:
                desc = getattr(s, 'description', '').upper()
                path = getattr(s, 'path', '').upper()
                if not _is_invalid_equity(s, clean):
                    if any(k in path for k in ["COMMODIT", "ENERGI", "ENERGY", "OIL", "FUTURES", "CFD", "SPOT"]):
                        if target_term in desc or "CRUDE OIL" in desc or "SPOT WTI" in desc or "LIGHT SWEET" in desc:
                            return s.name

        # Fallback 3c: General prefix match for forex (e.g. EURUSD.r, EURUSDm)
        base_sym = clean.split(".")[0]
        for s in all_syms:
            if not _is_invalid_equity(s, clean):
                if s.name.upper().startswith(base_sym):
                    return s.name

    return symbol


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

    term_path = _get_terminal_path_for_server(server, str(user.get("terminal_path") or ""))

    logger.info(f"Worker init: {user_name} (#{login}) on {server} via terminal [{term_path}]")

    try:
        mt5.shutdown()
    except Exception:
        pass

    # Pass login credentials directly into initialize() to avoid mt5.login() which
    # persists the account to the terminal profile on disk (causing cross-terminal bleed).
    init_kwargs = {
        "login": int(login),
        "password": str(password),
        "server": str(server),
        "timeout": 15000
    }
    if term_path:
        init_kwargs["path"] = term_path

    init_ok = False
    for attempt in range(2):
        if mt5.initialize(**init_kwargs):
            init_ok = True
            break
        err = mt5.last_error()
        err_code = err[0] if isinstance(err, (list, tuple)) and len(err) > 0 else err
        if attempt == 0 and err_code == -10005:
            logger.warning(f"Worker MT5 IPC timeout on attempt 1 for {user_name}. Retrying in 1.5s...")
            time.sleep(1.5)
        else:
            break

    if not init_ok:
        err = mt5.last_error()
        logger.error(f"❌ Worker MT5 initialize failed for {user_name}: {err}")
        return {"status": "FAILED", "error": f"INIT_FAILED_{err}"}

    acc = mt5.account_info()

    if not acc or acc.login != login:
        logger.error(f"❌ Account mismatch in worker! Target #{login}, but connected to #{getattr(acc, 'login', 'None')}. "
                     f"Credentials may be wrong or terminal may not be running.")
        mt5.shutdown()
        return {"status": "FAILED", "error": "ACCOUNT_MISMATCH"}

    logger.info(f"✅ Worker logged in: {acc.name} (#{acc.login}) on {acc.server} | Balance: ${acc.balance:,.2f}")

    # ── Secondary Account Daily Drawdown Check ────────────────────────
    # Dynamically scales to subscriber's account tier:
    # e.g., $10k -> $450, $100k -> $4,500, $1M -> $45,000 (4.5%)
    from core.guardrail import detect_account_tier
    from datetime import timedelta
    tier = detect_account_tier(acc.balance)
    max_user_loss = (float(user.get("max_daily_drawdown_pct", 4.5)) / 100.0) * tier
    now_utc = datetime.now(timezone.utc)
    today_start = (now_utc + timedelta(hours=2)).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=2)
    deals = mt5.history_deals_get(today_start, now_utc)
    today_closed = sum(d.profit + d.commission + d.swap for d in (deals or []))
    today_floating = acc.equity - acc.balance
    today_pnl = today_closed + today_floating

    if today_pnl <= -max_user_loss:
        logger.warning(
            f"🛑 DAILY DRAWDOWN CEILING: Subscriber #{login} ({user_name}) reached daily loss limit "
            f"(${abs(today_pnl):,.2f} >= ${max_user_loss:,.2f} [4.5% of ${tier:,.0f} tier]). Blocking new order."
        )
        mt5.shutdown()
        return {"status": "SKIPPED", "reason": "SUBSCRIBER_DAILY_DRAWDOWN_LIMIT_REACHED"}

    # Resolve broker-specific symbol alias (e.g. USOIL.cash -> WTI)
    actual_symbol = resolve_symbol_for_server(symbol, mt5)

    # Check if position already open on this account (Deduplication)
    existing_pos = mt5.positions_get(symbol=actual_symbol)
    if existing_pos:
        for p in existing_pos:
            if (signal_type == "BUY" and p.type == 0) or (signal_type == "SELL" and p.type == 1):
                logger.warning(f"🛑 DEDUP: {actual_symbol} {signal_type} already open on #{login} (Ticket #{p.ticket}). Blocking duplicate.")
                mt5.shutdown()
                return {"status": "SKIPPED", "ticket": p.ticket, "reason": "ALREADY_OPEN"}

    if not mt5.symbol_select(actual_symbol, True):
        from core.symbol_guard import is_commodity
        err_msg = f"SYMBOL_NOT_FOUND: {actual_symbol} not available on {server}"
        if "MetaQuotes" in server and is_commodity(symbol):
            err_msg = f"BROKER_UNSUPPORTED: {actual_symbol} is not offered on MetaQuotes-Demo. Connect to FTMO or a commodity broker account to trade Oil."
        logger.error(f"❌ {err_msg}")
        mt5.shutdown()
        return {"status": "FAILED", "error": err_msg}

    # Poll briefly for live quotes if newly selected
    tick = None
    for _ in range(35):
        tick = mt5.symbol_info_tick(actual_symbol)
        if tick and (tick.ask > 0 or tick.bid > 0):
            break
        time.sleep(0.1)

    if not tick or (tick.ask <= 0 and tick.bid <= 0):
        logger.error(f"❌ No live tick for {actual_symbol}")
        mt5.shutdown()
        return {"status": "FAILED", "error": "NO_TICK"}

    price = tick.ask if signal_type == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal_type == "BUY" else mt5.ORDER_TYPE_SELL

    # Dynamic 0.5% risk lot size calculation
    risk_value = float(user.get("risk_value", 0.5))
    risk_amount = acc.balance * (risk_value / 100.0)

    s_info = mt5.symbol_info(actual_symbol)
    default_pips = 0.28 if "JPY" in actual_symbol else 0.0028
    if sl <= 0:
        price_dist = default_pips
    else:
        price_dist = abs(price - sl)
        if price_dist > (price * 0.05):
            price_dist = default_pips

    # Calculate loss per lot using native MT5 order_calc_profit FIRST.
    # This automatically accounts for broker contract size (e.g. 100 oz Gold, 1000 bbl Oil),
    # digits, point value, and quote currency conversions with 100% accuracy.
    loss_per_lot = None
    target_sl = (price - price_dist) if signal_type == "BUY" else (price + price_dist)
    try:
        profit_1lot = mt5.order_calc_profit(order_type, actual_symbol, 1.0, price, target_sl)
        if profit_1lot is not None and abs(profit_1lot) > 0:
            loss_per_lot = abs(profit_1lot)
            logger.debug(f"Native order_calc_profit for 1.0 lot {actual_symbol}: ${loss_per_lot:.2f}")
    except Exception as _ce:
        logger.warning(f"order_calc_profit failed for {actual_symbol}: {_ce}")

    if not loss_per_lot or loss_per_lot <= 0:
        # Fallback to tick-math formula if order_calc_profit is unavailable
        tick_size = getattr(s_info, 'trade_tick_size', 0.00001) or 0.00001
        tick_val = getattr(s_info, 'trade_tick_value', 1.0) or 1.0
        dist_in_ticks = price_dist / tick_size if tick_size > 0 else 0
        loss_per_lot = dist_in_ticks * tick_val if (dist_in_ticks > 0 and tick_val > 0) else 1.0
        logger.debug(f"Fallback tick-math loss_per_lot for 1.0 lot {actual_symbol}: ${loss_per_lot:.2f}")

    raw_lots = risk_amount / loss_per_lot

    # ── Master-Proportional Sanity Cap ────────────────────────────────
    # Prevents any broker tick/contract calculation discrepancy from opening oversized trades.
    # If Master placed 0.02 lots on $10k, Secondary ($100k) should place ~0.20 lots, NOT 1.61!
    master_vol = float(signal_row.get("master_volume") or signal_row.get("master_lots") or signal_row.get("volume") or 0.0)
    master_bal = float(signal_row.get("master_balance") or 10000.0)
    if master_vol > 0 and master_bal > 0:
        expected_scale = acc.balance / master_bal
        expected_vol = master_vol * expected_scale
        # Cap at 1.5x expected proportional volume
        if raw_lots > expected_vol * 1.5:
            logger.warning(
                f"⚠️ LOT SANITY CAP on #{login}: Calculated raw lots ({raw_lots:.2f}) exceeds 1.5x master-proportional volume "
                f"({expected_vol:.2f} based on Master {master_vol:.2f} lots). Clamping to {expected_vol:.2f}."
            )
            raw_lots = expected_vol

    step = getattr(s_info, 'volume_step', 0.01) or 0.01
    volume = round(raw_lots / step) * step
    vol_min = getattr(s_info, 'volume_min', 0.01) or 0.01
    vol_max = getattr(s_info, 'volume_max', 100.0) or 100.0
    volume = max(vol_min, min(vol_max, volume))

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
        "symbol":       actual_symbol,
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

    logger.info(f"📤 Placing order: {actual_symbol} (original: {symbol}) {signal_type} | Lots: {volume:.2f} (Risk: ${risk_amount:.2f}) | Filling: {filling_type}")
    res = mt5.order_send(request)

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"  ✅ SUCCESS: Placed {actual_symbol} {signal_type} {volume:.2f} lots! Ticket #{res.order}")
        mt5.shutdown()
        return {"status": "SUCCESS", "ticket": res.order, "volume": volume, "symbol": actual_symbol}
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

    term_path = _get_terminal_path_for_server(server, str(user.get("terminal_path") or ""))

    try:
        mt5.shutdown()
    except Exception:
        pass

    init_kwargs = {"login": login, "password": password, "server": server, "timeout": 15000}
    if term_path:
        init_kwargs["path"] = term_path

    if not mt5.initialize(**init_kwargs):
        return {"status": "FAILED", "error": "INIT_FAILED"}

    actual_symbol = resolve_symbol_for_server(symbol, mt5)
    positions = mt5.positions_get(symbol=actual_symbol)
    if not positions and actual_symbol != symbol:
        positions = mt5.positions_get(symbol=symbol)

    if not positions:
        mt5.shutdown()
        return {"status": "NO_OPEN_POSITIONS"}

    closed_tickets = []
    failed_tickets = []
    for pos in positions:
        ticket = pos.ticket
        vol = pos.volume
        pos_sym = pos.symbol
        calc_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos_sym)
        if not tick:
            failed_tickets.append({"ticket": ticket, "symbol": pos_sym, "reason": "NO_TICK"})
            continue
        price = tick.bid if calc_type == mt5.ORDER_TYPE_SELL else tick.ask

        s_info = mt5.symbol_info(pos_sym)
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
            "symbol": pos_sym,
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
            logger.info(f"✅ Closed ticket #{ticket} ({pos_sym}) on #{login}")
            closed_tickets.append(ticket)
        else:
            retcode = res.retcode if res else -1
            comment = res.comment if res else "No response"
            logger.warning(f"❌ Failed to close #{ticket} ({pos_sym}) on #{login}: {comment} (Code: {retcode})")
            failed_tickets.append({"ticket": ticket, "retcode": retcode, "error": comment})

    mt5.shutdown()
    return {"status": "CLOSED" if closed_tickets else "FAILED", "tickets": closed_tickets, "failed": failed_tickets}


def _worker_close_all(user: dict, reason: str = "") -> dict:
    """Executed inside an ISOLATED worker subprocess to close ALL positions on that account."""
    import MetaTrader5 as mt5

    login = int(user["mt5_login"])
    password = str(user["mt5_password"])
    server = str(user["mt5_server"])
    user_name = user.get("name", f"User_{login}")
    term_path = _get_terminal_path_for_server(server, str(user.get("terminal_path") or ""))

    try:
        mt5.shutdown()
    except Exception:
        pass

    init_kwargs = {"login": login, "password": password, "server": server, "timeout": 15000}
    if term_path:
        init_kwargs["path"] = term_path

    if not mt5.initialize(**init_kwargs):
        return {"status": "FAILED", "error": "INIT_FAILED"}

    positions = mt5.positions_get()
    if not positions:
        mt5.shutdown()
        return {"status": "NO_OPEN_POSITIONS", "closed": []}

    closed_tickets = []
    failed_tickets = []
    for pos in positions:
        ticket = pos.ticket
        vol = pos.volume
        pos_sym = pos.symbol
        calc_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos_sym)
        if not tick:
            failed_tickets.append({"ticket": ticket, "symbol": pos_sym, "reason": "NO_TICK"})
            continue
        price = tick.bid if calc_type == mt5.ORDER_TYPE_SELL else tick.ask

        s_info = mt5.symbol_info(pos_sym)
        filling_type = mt5.ORDER_FILLING_IOC
        if s_info:
            if (s_info.filling_mode & 2) != 0:
                filling_type = mt5.ORDER_FILLING_IOC
            elif (s_info.filling_mode & 1) != 0:
                filling_type = mt5.ORDER_FILLING_FOK
            else:
                filling_type = mt5.ORDER_FILLING_RETURN

        comment = f"ForexAlert {reason}" if reason else "ForexAlert Close All"
        comment = comment[:31]

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos_sym,
            "volume": vol,
            "type": calc_type,
            "position": ticket,
            "price": price,
            "deviation": 30,
            "magic": 20260622,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Closed position #{ticket} ({pos_sym}) on #{login}")
            closed_tickets.append(ticket)
        else:
            retcode = res.retcode if res else -1
            comment_err = res.comment if res else "No response"
            logger.warning(f"❌ Failed to close ticket #{ticket} ({pos_sym}) on #{login}: {comment_err} (Code: {retcode})")
            failed_tickets.append({"ticket": ticket, "symbol": pos_sym, "retcode": retcode, "error": comment_err})

    mt5.shutdown()
    return {"status": "PROCESSED", "closed": closed_tickets, "failed": failed_tickets}


def _worker_sync(user: dict, master_symbols: list) -> dict:
    """
    Reconcile secondary account positions with master account.
    Any position open on this account whose symbol is NOT in master_symbols
    (resolved to this broker's naming convention) will be liquidated.
    """
    import MetaTrader5 as mt5

    login = int(user["mt5_login"])
    password = str(user["mt5_password"])
    server = str(user["mt5_server"])
    user_name = user.get("name", f"User_{login}")
    term_path = _get_terminal_path_for_server(server, str(user.get("terminal_path") or ""))

    try:
        mt5.shutdown()
    except Exception:
        pass

    init_kwargs = {"login": login, "password": password, "server": server, "timeout": 15000}
    if term_path:
        init_kwargs["path"] = term_path

    if not mt5.initialize(**init_kwargs):
        return {"status": "FAILED", "error": "INIT_FAILED"}

    # Build the set of allowed symbols on this secondary account
    allowed_symbols = set()
    for ms in master_symbols:
        resolved = resolve_symbol_for_server(ms, mt5)
        allowed_symbols.add(resolved.upper())
        allowed_symbols.add(ms.upper())

    positions = mt5.positions_get()
    if not positions:
        mt5.shutdown()
        return {"status": "SYNCED", "closed": [], "open_count": 0}

    # ── Secondary Account Daily Drawdown Check ────────────────────────
    # If subscriber's own daily loss reaches 4.5% of their tier ($4,500 on $100k),
    # trigger emergency liquidation for this account to protect the challenge.
    from core.guardrail import detect_account_tier
    from datetime import timedelta
    acc = mt5.account_info()
    if acc:
        tier = detect_account_tier(acc.balance)
        max_user_loss = (float(user.get("max_daily_drawdown_pct", 4.5)) / 100.0) * tier
        now_utc = datetime.now(timezone.utc)
        today_start = (now_utc + timedelta(hours=2)).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=2)
        deals = mt5.history_deals_get(today_start, now_utc)
        today_closed = sum(d.profit + d.commission + d.swap for d in (deals or []))
        today_floating = acc.equity - acc.balance
        today_pnl = today_closed + today_floating

        if today_pnl <= -max_user_loss and positions:
            logger.critical(
                f"🚨 SUBSCRIBER DRAWDOWN KILL SWITCH: Account #{login} ({user_name}) reached daily loss limit "
                f"(${abs(today_pnl):,.2f} >= ${max_user_loss:,.2f} [4.5% of ${tier:,.0f} tier]). Liquidating all positions!"
            )
            allowed_symbols = set()  # Force all open positions to be liquidated

    migration_cutoff = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc).timestamp()
    closed_tickets = []
    failed_tickets = []
    for pos in positions:
        sym_upper = pos.symbol.upper()
        if sym_upper not in allowed_symbols:
            # Option B Guard: Preserve existing legacy positions opened prior to Master migration
            # if they have hard broker-side SL/TP protection.
            if getattr(pos, 'time', 0) < migration_cutoff and (pos.sl > 0 or pos.tp > 0):
                logger.info(f"  ✓ Preserving legacy position #{pos.ticket} ({pos.symbol}) on #{login} with SL={pos.sl}/TP={pos.tp} to run naturally (Option B).")
                continue

            logger.warning(f"🔄 DESYNC DETECTED on #{login} ({user_name}): Position #{pos.ticket} ({pos.symbol}) not present on Master. Liquidating...")
            ticket = pos.ticket
            vol = pos.volume
            calc_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                failed_tickets.append({"ticket": ticket, "symbol": pos.symbol, "reason": "NO_TICK"})
                continue
            price = tick.bid if calc_type == mt5.ORDER_TYPE_SELL else tick.ask

            s_info = mt5.symbol_info(pos.symbol)
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
                "symbol": pos.symbol,
                "volume": vol,
                "type": calc_type,
                "position": ticket,
                "price": price,
                "deviation": 30,
                "magic": 20260622,
                "comment": "ForexAlert Sync Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"  ✅ Synced: Liquidated orphan #{ticket} ({pos.symbol}) on #{login}")
                closed_tickets.append(ticket)
            else:
                retcode = res.retcode if res else -1
                comment_err = res.comment if res else "No response"
                logger.warning(f"  ❌ Sync liquidation failed for #{ticket} ({pos.symbol}): {comment_err} (Code: {retcode})")
                failed_tickets.append({"ticket": ticket, "symbol": pos.symbol, "retcode": retcode, "error": comment_err})
        else:
            logger.debug(f"  ✓ Position #{pos.ticket} ({pos.symbol}) matches Master. Keeping open.")

    mt5.shutdown()
    remaining = len(positions) - len(closed_tickets)
    return {"status": "SYNCED", "closed": closed_tickets, "failed": failed_tickets, "open_count": remaining}


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
            "--user-json", json.dumps(dict(user), default=str),
            "--signal-json", json.dumps(dict(signal_row), default=str)
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(PROJECT_ROOT))
            out = proc.stdout.strip()
            if proc.returncode == 0:
                logger.info(f"  Worker Output for {user_name}:\n{out}")
                results[user_name] = "SUCCESS"
                mark_last_trade(user_id)
            else:
                err_detail = proc.stderr.strip()
                if not err_detail:
                    for line in reversed(out.splitlines()):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                res_obj = json.loads(line)
                                err_detail = res_obj.get("error") or res_obj.get("details") or res_obj.get("reason") or ""
                                if err_detail:
                                    break
                            except Exception:
                                pass
                if not err_detail:
                    err_detail = out.splitlines()[-1] if out.splitlines() else f"Exit code {proc.returncode}"
                logger.error(f"  Worker error for {user_name} (Exit code {proc.returncode}): {err_detail}\n{out}")
                results[user_name] = f"ERROR: {err_detail}"
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ Worker timed out for {user_name}")
            results[user_name] = "TIMEOUT"
        except Exception as _we:
            logger.error(f"  ❌ Worker exception for {user_name}: {_we}")
            results[user_name] = f"EXCEPTION: {_we}"

        # Brief pause between sequential workers sharing terminal to ensure IPC pipe resets cleanly
        time.sleep(0.8)

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
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(PROJECT_ROOT))
            if proc.returncode == 0:
                results[user_name] = "CLOSED"
            else:
                results[user_name] = f"ERROR: {proc.stderr.strip()}"
        except Exception as _ce:
            results[user_name] = f"EXCEPTION: {_ce}"

    return results


def close_all_positions_for_all_users(reason: str = "") -> dict:
    """Close ALL positions across all enabled users via isolated subprocesses."""
    from core.user_accounts import get_enabled_users

    users = get_enabled_users()
    if not users:
        return {}

    logger.info(f"🌐 Multi-Executor: Liquidating ALL positions for {len(users)} user(s) (Reason: {reason or 'None'})")
    results = {}

    for user in users:
        user_name = user["name"]
        cmd = [
            sys.executable,
            "-X", "utf8",
            "-m", "scripts.multi_executor",
            "--worker-close-all",
            "--user-json", json.dumps(dict(user)),
            "--reason", reason or ""
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(PROJECT_ROOT))
            if proc.returncode == 0:
                results[user_name] = "SUCCESS"
            else:
                results[user_name] = f"ERROR: {proc.stderr.strip()}"
        except Exception as _ce:
            results[user_name] = f"EXCEPTION: {_ce}"

    return results


def sync_positions_with_master(master_symbols: list) -> dict:
    """
    Synchronize open positions on all secondary accounts with the master account.
    Any positions on secondary accounts not present on the master account will be closed.
    """
    from core.user_accounts import get_enabled_users

    users = get_enabled_users()
    if not users:
        return {}

    results = {}
    for user in users:
        user_name = user["name"]
        cmd = [
            sys.executable,
            "-X", "utf8",
            "-m", "scripts.multi_executor",
            "--worker-sync",
            "--user-json", json.dumps(dict(user)),
            "--master-symbols-json", json.dumps(master_symbols)
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(PROJECT_ROOT))
            if proc.returncode == 0:
                results[user_name] = "SYNCED"
            else:
                results[user_name] = f"ERROR: {proc.stderr.strip()}"
        except Exception as _se:
            results[user_name] = f"EXCEPTION: {_se}"

    return results


def _worker_test_connection(user: dict) -> dict:
    """
    Test MT5 connection inside an ISOLATED worker subprocess.
    Verifies credentials, server connection, balance, leverage,
    and checks instrument coverage (Forex, Gold, Oil).
    """
    import MetaTrader5 as mt5

    try:
        login = int(user.get("mt5_login") or 0)
    except (ValueError, TypeError):
        return {"status": "FAILED", "error": "INVALID_LOGIN", "details": "Account login number must be numeric digits."}

    password = str(user.get("mt5_password") or "")
    server = str(user.get("mt5_server") or "")
    user_term = str(user.get("terminal_path") or "")

    if not login or not password or not server:
        return {"status": "FAILED", "error": "MISSING_FIELDS", "details": "Please fill in Login, Password, and Server."}

    term_path = _get_terminal_path_for_server(server, user_term)

    try:
        mt5.shutdown()
    except Exception:
        pass

    init_kwargs = {
        "login": int(login),
        "password": str(password),
        "server": str(server),
        "timeout": 12000
    }
    if term_path:
        init_kwargs["path"] = term_path

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        err_code = err[0] if isinstance(err, (list, tuple)) and len(err) > 0 else err
        err_msg = err[1] if isinstance(err, (list, tuple)) and len(err) > 1 else str(err)
        return {
            "status": "FAILED",
            "error": f"INIT_FAILED_{err_code}",
            "details": f"Connection failed: {err_msg} (Code {err_code}). Check server name spelling, account number, or password."
        }

    acc = mt5.account_info()
    if not acc or acc.login != login:
        mt5.shutdown()
        return {
            "status": "FAILED",
            "error": "ACCOUNT_MISMATCH",
            "details": f"Connected to terminal, but account #{login} was not authorized. Verify credentials."
        }

    # Test Gold and Oil symbol resolution
    gold_res = resolve_symbol_for_server("XAUUSD", mt5)
    oil_res = resolve_symbol_for_server("USOIL.cash", mt5)

    gold_ok = False
    if gold_res:
        mt5.symbol_select(gold_res, True)
        tick = mt5.symbol_info_tick(gold_res)
        gold_ok = tick is not None

    oil_ok = False
    if oil_res:
        mt5.symbol_select(oil_res, True)
        tick = mt5.symbol_info_tick(oil_res)
        oil_ok = tick is not None

    # Query active positions on this account
    positions = mt5.positions_get()
    open_positions = []
    if positions:
        for p in positions:
            open_positions.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": round(float(p.volume), 2),
                "price_open": round(float(p.price_open), 5),
                "price_current": round(float(p.price_current), 5),
                "sl": round(float(p.sl), 5),
                "tp": round(float(p.tp), 5),
                "profit": round(float(p.profit), 2),
                "swap": round(float(p.swap), 2),
                "magic": p.magic,
                "comment": p.comment,
            })

    res = {
        "status": "SUCCESS",
        "login": acc.login,
        "name": acc.name,
        "company": getattr(acc, "company", "") or server,
        "server": acc.server,
        "balance": acc.balance,
        "equity": acc.equity,
        "currency": acc.currency,
        "leverage": acc.leverage,
        "terminal_path": term_path or "Default MT5",
        "gold_symbol": gold_res,
        "gold_supported": gold_ok,
        "oil_symbol": oil_res,
        "oil_supported": oil_ok,
        "open_positions": open_positions,
        "open_positions_count": len(open_positions),
    }
    mt5.shutdown()
    return res


def get_account_live_positions(user: dict) -> list:
    """Fetch open positions for an account using isolated worker subprocess."""
    res = test_mt5_account_connection(user)
    if res.get("status") == "SUCCESS":
        return res.get("open_positions", [])
    return []


def test_mt5_account_connection(user: dict) -> dict:
    """
    Public helper to test an MT5 account connection safely in an isolated subprocess.
    """
    cmd = [
        sys.executable,
        "-X", "utf8",
        "-m", "scripts.multi_executor",
        "--worker-test",
        "--user-json", json.dumps(dict(user), default=str)
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            cwd=str(PROJECT_ROOT)
        )
        out = proc.stdout.strip()
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except Exception:
                    pass
        if proc.returncode != 0:
            return {"status": "FAILED", "error": "WORKER_ERROR", "details": proc.stderr.strip() or out}
        return {"status": "FAILED", "error": "NO_OUTPUT", "details": out}
    except subprocess.TimeoutExpired:
        return {"status": "FAILED", "error": "TIMEOUT", "details": "Connection test timed out after 20 seconds."}
    except Exception as e:
        return {"status": "FAILED", "error": "EXCEPTION", "details": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-exec", action="store_true", help="Run order execution worker")
    parser.add_argument("--worker-close", action="store_true", help="Run close worker")
    parser.add_argument("--worker-close-all", action="store_true", help="Run close-all worker")
    parser.add_argument("--worker-sync", action="store_true", help="Run sync positions worker")
    parser.add_argument("--worker-test", action="store_true", help="Run test connection worker")
    parser.add_argument("--user-json", type=str, help="User credentials JSON string")
    parser.add_argument("--signal-json", type=str, help="Signal details JSON string")
    parser.add_argument("--symbol", type=str, help="Symbol to close")
    parser.add_argument("--reason", type=str, default="", help="Reason for closure")
    parser.add_argument("--master-symbols-json", type=str, default="[]", help="JSON list of open master symbols")
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

    elif args.worker_close_all and args.user_json:
        user_data = json.loads(args.user_json)
        res = _worker_close_all(user_data, args.reason)
        print(json.dumps(res))
        sys.exit(0 if res.get("status") in ("PROCESSED", "NO_OPEN_POSITIONS") else 1)

    elif args.worker_sync and args.user_json:
        user_data = json.loads(args.user_json)
        master_syms = json.loads(args.master_symbols_json or "[]")
        res = _worker_sync(user_data, master_syms)
        print(json.dumps(res))
        sys.exit(0 if res.get("status") == "SYNCED" else 1)

    elif args.worker_test and args.user_json:
        user_data = json.loads(args.user_json)
        res = _worker_test_connection(user_data)
        print(json.dumps(res))
        sys.exit(0 if res.get("status") == "SUCCESS" else 1)

