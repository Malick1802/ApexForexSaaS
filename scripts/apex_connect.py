import sqlite3
import yaml
import time
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mt5_connector import get_mt5
from scripts.multi_executor import execute_signal_for_all_users

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
GHOST_TRADES_CSV = LOG_DIR / "ghost_trades.csv"

logger = logging.getLogger("ApexConnect")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "apex_connect.log", encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - APEX_CONNECT - %(levelname)s - %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s - APEX_CONNECT - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

# Config Path
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DB_PATH = PROJECT_ROOT / "signals.db"

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def calculate_lots(symbol, risk_type, risk_value, sl_price, entry_price, config):
    """
    Calculate lot size from risk settings. Supports three modes:
      - 'fixed'     : risk_value is a direct lot size (e.g. 0.01)
      - 'fixed_usd' : risk_value is a fixed dollar amount per trade (e.g. 1.0)
      - 'percent'   : risk_value is a % of account balance (e.g. 0.20)
    Ensures margin-safe execution in all modes.
    """
    if risk_type == 'fixed':
        return float(risk_value)

    try:
        _mt5 = get_mt5()
        if not _mt5:
            logger.error("🚫 MT5 connection unavailable for lot calculation.")
            return 0.01

        mt5_conf = config.get('mt5', {})
        max_leverage = mt5_conf.get('max_trade_leverage', 30)

        account_info = _mt5.account_info()
        if not account_info:
            return 0.01

        symbol_info = _mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01

        # ── Determine dollar risk amount ─────────────────────────────────────
        if risk_type == 'fixed_usd':
            # Exact fixed dollar amount per trade regardless of balance
            risk_amount = float(risk_value)
            logger.info(f"💵 Fixed USD risk mode: ${risk_amount:.2f} per trade")
        else:
            # percent mode: risk_value % of current balance
            risk_amount = account_info.balance * (risk_value / 100.0)
            logger.info(f"📊 Percent risk mode: {risk_value}% of ${account_info.balance:.2f} = ${risk_amount:.2f}")

        # ── 1. Calculate Risk-Based Lots ─────────────────────────────────────
        loss_per_lot = None
        try:
            order_type = _mt5.ORDER_TYPE_BUY if sl_price < entry_price else _mt5.ORDER_TYPE_SELL
            profit_1lot = _mt5.order_calc_profit(order_type, symbol, 1.0, entry_price, sl_price)
            if profit_1lot is not None and abs(profit_1lot) > 0:
                loss_per_lot = abs(profit_1lot)
        except Exception:
            pass

        if not loss_per_lot or loss_per_lot <= 0:
            tick_size = symbol_info.trade_tick_size or 0.00001
            tick_value = symbol_info.trade_tick_value or 1.0
            price_dist = abs(entry_price - sl_price)
            dist_in_ticks = price_dist / tick_size
            if dist_in_ticks <= 0 or tick_value <= 0:
                return 0.01
            loss_per_lot = dist_in_ticks * tick_value

        risk_lots = risk_amount / loss_per_lot

        # ── 2. Calculate Margin-Limited Maximum (Currency-Agnostic) ──────────
        margin_per_lot = _mt5.order_calc_margin(_mt5.ORDER_TYPE_BUY, symbol, 1.0, entry_price)

        if not margin_per_lot:
            # Fallback
            notional_per_lot = entry_price * symbol_info.trade_contract_size
            margin_per_lot = notional_per_lot / max_leverage

        # Max safe lots (using 90% of buying power)
        max_margin_lots = (account_info.balance * 0.9) / margin_per_lot

        # ── 3. Final Lot Size (Smallest of Risk vs Margin) ───────────────────
        final_lots = min(risk_lots, max_margin_lots)

        # Normalize to Volume Step
        step = symbol_info.volume_step
        final_lots = round(final_lots / step) * step

        return max(symbol_info.volume_min, min(symbol_info.volume_max, final_lots))

    except Exception as e:
        logger.error(f"ApexConnect lot calculation failed: {e}")
        return 0.01

def place_trade(signal_row, config):
    if not isinstance(signal_row, dict):
        try:
            signal_row = dict(signal_row)
        except Exception:
            pass
    symbol = signal_row['symbol']
    signal_type = signal_row['signal']
    sl = signal_row['sl_price']
    tp = signal_row['tp_price']
    entry_est = signal_row['price_at_signal'] # Estimated entry
    regime = signal_row.get('regime', 'NORMAL') if isinstance(signal_row, dict) else 'NORMAL'

    # ── COMMODITY / BLOCKED SYMBOL / DIRECTIONAL SAFETY GATE ──
    from core.symbol_guard import is_symbol_blocked, is_direction_blocked
    if is_symbol_blocked(symbol):
        logger.critical(f"🛑 COMMODITY SHIELD: Symbol {symbol} is a blacklisted commodity. Refusing MT5 order execution entirely!")
        return None
    if is_direction_blocked(symbol, signal_type):
        logger.critical(f"🛑 DIRECTIONAL SHIELD: {symbol} {signal_type} is blacklisted by directional shield. Refusing MT5 order execution!")
        return None
    
    mt5_conf = config.get('mt5', {})
    risk_type = mt5_conf.get('risk_type', 'fixed')
    risk_val = mt5_conf.get('risk_value', 0.01)

    _mt5 = get_mt5()
    if not _mt5:
        logger.error("🚫 MT5 Connection lost while preparing trade.")
        return None

    # ── MAX CONCURRENT TRADES RISK SHIELD ──
    max_open = mt5_conf.get('max_open_trades', 9)
    if max_open > 0:
        open_positions = _mt5.positions_total()
        if open_positions >= max_open:
            logger.critical(f"🛑 RISK SHIELD: Maximum open trades reached ({open_positions}/{max_open}). Skipping order execution.")
            return None

    # Prepare info
    symbol_info = _mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"{symbol} not found in MT5")
        return None

    # Check/Select symbol first so we can get a valid tick
    if not _mt5.symbol_select(symbol, True):
        logger.error(f"❌ Symbol {symbol} not visible in MT5.")
        return None

    # Get live price for accurate lot calculation
    tick = _mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"❌ Failed to get tick for {symbol}")
        return None

    live_entry = tick.ask if signal_type == 'BUY' else tick.bid

    # Calculate Lots using LIVE entry price (not 0.0) for correct SL distance
    volume = calculate_lots(symbol, risk_type, risk_val, sl, live_entry, config)
    logger.info(f"💰 Risk calc: {risk_val}% | Entry: {live_entry} | SL: {sl} | Lots: {volume}")

    # Determine filling mode
    filling_type = _mt5.ORDER_FILLING_FOK
    symbol_info = _mt5.symbol_info(symbol)
    if symbol_info:
        if (symbol_info.filling_mode & 2) != 0:
            filling_type = _mt5.ORDER_FILLING_IOC
        elif (symbol_info.filling_mode & 1) != 0:
            filling_type = _mt5.ORDER_FILLING_FOK

    # Build Trade Request (tick already fetched above for lot calculation)
    price = live_entry
    ot_type = _mt5.ORDER_TYPE_BUY if signal_type == 'BUY' else _mt5.ORDER_TYPE_SELL

    request = {
        "action": _mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": ot_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": 20240401,
        "comment": f"Apex {regime}",
        "type_time": _mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    # ── NEWS/WEEKEND FILTER ──
    # If the signal is being placed during a filtered window (e.g. within 1 hour of Friday close), block it.
    from core.market_hours import is_friday_trade_entry_allowed
    allowed, reason = is_friday_trade_entry_allowed()
    if not allowed:
        logger.warning(f"🚫 BLOCKED: {reason}. No new trades for {symbol}.")
        return None

    # Shadow Mode Check
    execute = config.get('trading', {}).get('execute_trades', True)
    if not execute:
        logger.info(f"👻 SHADOW MODE: Skipping {symbol} {signal_type} execution.")
        
        # Log to Ghost Trades CSV for audit
        try:
            write_header = not GHOST_TRADES_CSV.exists()
            with open(GHOST_TRADES_CSV, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write("timestamp,symbol,signal,confidence,price,sl,tp,regime,vix_proxy,yield_slope,adx,atr_zscore\n")
                
                # Extract meta-data
                regime = signal_row.get('regime', 'UNKNOWN')
                vix = signal_row.get('vix_proxy', 0.0)
                slope = signal_row.get('yield_slope', 0.0)
                adx = signal_row.get('adx', 0.0)
                atr_z = signal_row.get('atr_zscore', 0.0)
                conf = signal_row.get('confidence', 0.0)
                
                f.write(f"{datetime.now().isoformat()},{symbol},{signal_type},{conf:.4f},{price},{sl},{tp},{regime},{vix:.4f},{slope:.4f},{adx:.1f},{atr_z:.2f}\n")
        except Exception as e:
            logger.error(f"Failed to write ghost trade: {e}")
            
        print(f"MT5_SHADOW: {symbol} {signal_type}")
        return 888888  # Synthetic "Shadow" ticket
    
    logger.info(f"🚀 EXECUTING {symbol} {signal_type} | Lots: {volume} | SL: {sl} | TP: {tp}")
    result = _mt5.order_send(request)
    
    if result is None:
        logger.error("❌ MT5 Order result is None - Bridge issue?")
        return None

    if result.retcode == _mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ TRADE SUCCESS! Ticket: {result.order}")
        print(f"MT5_SUCCESS: {symbol} {signal_type} Ticket {result.order}")
        return result.order
    else:
        err_msg = f"❌ TRADE FAILED: {result.comment} (Code: {result.retcode})"
        logger.error(err_msg)
        print(f"MT5_ERROR: {symbol} {signal_type} -> {result.comment} ({result.retcode})")
        return None

def main_loop():
    logger.info("🔌 Apex Connect Bridge Started")
    
    _mt5 = get_mt5()
    if not _mt5:
        logger.critical("Failed to connect to MT5 Bridge")
        return

    _friday_close_done = False  # Guard: ensures Friday exit only fires once per session

    try:
        while True:
            # 1. Reload Config
            config = load_config()
            mt5_conf = config.get('mt5', {})
            enabled = mt5_conf.get('enabled', False)
            
            if not enabled:
                # logger.debug("Bridge paused...")
                time.sleep(5)
                continue
                
            # 2. Check Signals
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Fetch ACTIVE signals meant for trading that haven't been traded yet
                # We interpret NULL mt5_ticket as "Not yet processed by automation"
                query = """
                    SELECT * FROM signals 
                    WHERE outcome='ACTIVE' 
                    AND (mt5_ticket IS NULL OR mt5_ticket = '')
                    AND (is_hidden IS NULL OR is_hidden = 0)
                """
                cursor.execute(query)
                rows = cursor.fetchall()
                
                for row in rows:
                    sig_id = row['id']
                    symbol = row['symbol']
                    signal_type = row['signal']
                    logger.info(f"🔎 Found Pending Signal: {symbol} (ID: {sig_id})")
                    
                    # 1. Mark as 'Processing' immediately to prevent race conditions
                    cursor.execute("UPDATE signals SET mt5_ticket='0' WHERE id=?", (sig_id,))
                    conn.commit()
                    
                    # ── Dynamic Reversal Early Exit ──────────────────────────────────
                    if signal_type in ('BUY', 'SELL'):
                        try:
                            from core.reversal_guard import get_approved_reversal_pairs
                            approved_pairs = get_approved_reversal_pairs()
                            
                            if symbol in approved_pairs:
                                # Check if there is an active running position in MT5 for this symbol
                                _mt5_conn = get_mt5()
                                if _mt5_conn:
                                    open_positions = _mt5_conn.positions_get(symbol=symbol)
                                    # If there is a position, and it's in the opposite direction
                                    # pos.type: 0 = BUY, 1 = SELL
                                    is_reversal = False
                                    for pos in open_positions:
                                        if (signal_type == 'BUY' and pos.type == 1) or (signal_type == 'SELL' and pos.type == 0):
                                            is_reversal = True
                                            break
                                            
                                    if is_reversal:
                                        logger.info(f"🔄 Reversal detected for {symbol}! Closing existing active position first.")
                                        
                                        # 1. Close master account position
                                        from scripts.close_position import close_positions
                                        close_positions(symbol)
                                        
                                        # 2. Close positions for all copy-trading subscribers
                                        from scripts.multi_executor import close_signal_for_all_users
                                        close_results = close_signal_for_all_users(symbol)
                                        
                                        # 3. Notify subscribers on Telegram
                                        try:
                                            from core.telegram_alerts import _load_bot_token, _send
                                            bot_token = _load_bot_token()
                                            if bot_token:
                                                from core.user_accounts import get_enabled_users
                                                for u in get_enabled_users():
                                                    chat_id = u.get("telegram_chat_id", "")
                                                    if chat_id:
                                                        msg = (
                                                            f"🚪 <b>Trade Closed Early</b>\n"
                                                            f"Position on <b>{symbol}</b> was closed early because a trend reversal signal occurred. "
                                                            f"Expectancy logic auto-exited to secure profits or minimize drawdown."
                                                        )
                                                        _send(bot_token, chat_id, msg)
                                        except Exception as _te:
                                            logger.warning(f"Telegram reversal notification error: {_te}")
                                            
                        except Exception as _re:
                            logger.error(f"Error handling reversal early exit check: {_re}")
                    # ── End Dynamic Reversal Check ──────────────────────────────────
                    
                    # 2. Execute trade (Only for BUY/SELL)
                    if signal_type in ('BUY', 'SELL'):
                        ticket = place_trade(row, config)
                        # 2b. Broadcast to all registered subscriber accounts
                        try:
                            sig_data = dict(row)
                            if _mt5:
                                m_acc = _mt5.account_info()
                                if m_acc:
                                    sig_data['master_balance'] = m_acc.balance
                                if ticket and isinstance(ticket, int) and ticket > 0:
                                    m_pos = _mt5.positions_get(ticket=ticket)
                                    if m_pos:
                                        sig_data['master_volume'] = m_pos[0].volume
                            execute_signal_for_all_users(sig_data)
                        except Exception as _me:
                            logger.error(f"Multi-executor error: {_me}")
                    else:
                        # Log WAIT signals to CSV/Audit but don't send order
                        ticket = 'AUDIT'
                    
                    # 3. Update with final ticket or failure code
                    status_code = ticket if ticket else -1
                    cursor.execute("UPDATE signals SET mt5_ticket=? WHERE id=?", (status_code, sig_id))
                    conn.commit()
                
                # 3. Friday Auto-Exit (Prop Firm Safety: 30 minutes before market close)
                from core.market_hours import is_friday_auto_exit_time, get_ny_time
                ny_now = get_ny_time()
                # Reset guard at the start of each new day (so Monday–Thursday it stays False)
                if ny_now.weekday() != 4:
                    _friday_close_done = False

                # Friday 30 minutes before market close (16:30 New York time)
                if is_friday_auto_exit_time() and not _friday_close_done:
                    _friday_close_done = True
                    logger.info(f"🕒 Friday Auto-Exit Triggered ({ny_now.strftime('%Y-%m-%d %H:%M:%S %Z')}) — 30 min before close. Closing all open positions.")
                    positions = _mt5.positions_get()
                    if positions:
                        for p in positions:
                            logger.info(f"💾 Closing position: {p.symbol} (Ticket: {p.ticket})")
                            tick = _mt5.symbol_info_tick(p.symbol)
                            if not tick:
                                logger.error(f"Cannot get tick for {p.symbol} — skipping.")
                                continue
                            close_price = tick.bid if p.type == _mt5.POSITION_TYPE_BUY else tick.ask
                            close_type = _mt5.ORDER_TYPE_SELL if p.type == _mt5.POSITION_TYPE_BUY else _mt5.ORDER_TYPE_BUY
                            filling = _mt5.ORDER_FILLING_FOK
                            s_info = _mt5.symbol_info(p.symbol)
                            if s_info and (s_info.filling_mode & 2) != 0:
                                filling = _mt5.ORDER_FILLING_IOC
                            close_request = {
                                "action": _mt5.TRADE_ACTION_DEAL,
                                "symbol": p.symbol,
                                "volume": p.volume,
                                "type": close_type,
                                "position": p.ticket,
                                "price": close_price,
                                "deviation": 20,
                                "magic": 999000,
                                "comment": "Apex Friday Exit",
                                "type_time": _mt5.ORDER_TIME_GTC,
                                "type_filling": filling,
                            }
                            res = _mt5.order_send(close_request)
                            if res and res.retcode == _mt5.TRADE_RETCODE_DONE:
                                logger.info(f"✅ Closed {p.symbol} ticket {p.ticket} at {close_price}")
                            else:
                                comment = res.comment if res else "No response"
                                logger.error(f"❌ Failed to close {p.ticket}: {comment}")
                    else:
                        logger.info("No open positions found — nothing to close.")

                    try:
                        from scripts.multi_executor import close_all_positions_for_all_users
                        close_all_positions_for_all_users(reason="Friday Auto-Exit")
                    except Exception as _ace:
                        logger.error(f"Multi-user Friday close error: {_ace}")
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Apex Connect Stopped.")
    finally:
        _mt5.shutdown()

if __name__ == "__main__":
    main_loop()
