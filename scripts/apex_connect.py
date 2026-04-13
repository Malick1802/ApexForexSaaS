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
from core.core.mt5_connector import get_mt5

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
GHOST_TRADES_CSV = LOG_DIR / "ghost_trades.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - APEX_CONNECT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "apex_connect.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ApexConnect")

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
    Calculate lot size based on 0.5% Risk of BALANCE for GetLeveraged.com (1:30).
    Ensures margin-safe execution.
    """
    if risk_type == 'fixed':
        return float(risk_value)
    
    try:
        mt5_conf = config.get('mt5', {})
        max_leverage = mt5_conf.get('max_trade_leverage', 30)
        
        account_info = mt5.account_info()
        if not account_info:
            return 0.01

        # Calculate using BALANCE to protect trailing drawdown
        risk_amount = account_info.balance * (risk_value / 100.0)
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01

        # 1. Calculate Risk-Based Lots
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        price_dist = abs(entry_price - sl_price)
        dist_in_ticks = price_dist / tick_size

        if dist_in_ticks <= 0 or tick_value <= 0:
            return 0.01
            
        loss_per_lot = dist_in_ticks * tick_value
        risk_lots = risk_amount / loss_per_lot

        # 2. Calculate Margin-Limited Maximum (Currency-Agnostic)
        margin_per_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, entry_price)
        
        if not margin_per_lot:
            # Fallback
            notional_per_lot = entry_price * symbol_info.trade_contract_size
            margin_per_lot = notional_per_lot / max_leverage

        # Max safe lots (using 90% of buying power)
        max_margin_lots = (account_info.balance * 0.9) / margin_per_lot

        # 3. Final Lot Size (Smallest of Risk vs Margin)
        final_lots = min(risk_lots, max_margin_lots)
        
        # Normalize to Volume Step
        step = symbol_info.volume_step
        final_lots = round(final_lots / step) * step
        
        return max(symbol_info.volume_min, min(symbol_info.volume_max, final_lots))

    except Exception as e:
        logger.error(f"ApexConnect lot calculation failed: {e}")
        return 0.01

def place_trade(signal_row, config):
    symbol = signal_row['symbol']
    signal_type = signal_row['signal']
    sl = signal_row['sl_price']
    tp = signal_row['tp_price']
    entry_est = signal_row['price_at_signal'] # Estimated entry
    
    mt5_conf = config.get('mt5', {})
    risk_type = mt5_conf.get('risk_type', 'fixed')
    risk_val = mt5_conf.get('risk_value', 0.01)

    # Prepare info
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"{symbol} not found in MT5")
        return None

    # Calculate Lots
    volume = calculate_lots(symbol, risk_type, risk_val, sl, 0.0, config)
    
    _mt5 = get_mt5()
    if not _mt5:
        logger.error("🚫 MT5 Connection lost while preparing trade.")
        return None

    # Check/Select symbol
    if not _mt5.symbol_select(symbol, True):
        logger.error(f"❌ Symbol {symbol} not visible in MT5.")
        return None

    # Determine filling mode
    filling_type = _mt5.ORDER_FILLING_FOK
    symbol_info = _mt5.symbol_info(symbol)
    if symbol_info:
        if (symbol_info.filling_mode & 2) != 0:
            filling_type = _mt5.ORDER_FILLING_IOC
        elif (symbol_info.filling_mode & 1) != 0:
            filling_type = _mt5.ORDER_FILLING_FOK

    # Build Trade Request
    tick = _mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"❌ Failed to get tick for {symbol}")
        return None

    price = tick.ask if signal_type == 'BUY' else tick.bid
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
    # If the signal is being placed during a filtered window, block it.
    if datetime.now(timezone.utc).weekday() == 4: # Friday
        if datetime.now().hour >= 16: # 4PM (System local check)
             logger.warning(f"🚫 BLOCKED: Weekend approach. No new trades for {symbol}.")
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
                    # Setting ticket to '0' as a temporary placeholder (string for consistency)
                    cursor.execute("UPDATE signals SET mt5_ticket='0' WHERE id=?", (sig_id,))
                    conn.commit()
                    
                    # 2. Execute trade (Only for BUY/SELL)
                    if signal_type in ('BUY', 'SELL'):
                        ticket = place_trade(row, config)
                    else:
                        # Log WAIT signals to CSV/Audit but don't send order
                        ticket = 'AUDIT'
                    
                    # 3. Update with final ticket or failure code
                    status_code = ticket if ticket else -1
                    cursor.execute("UPDATE signals SET mt5_ticket=? WHERE id=?", (status_code, sig_id))
                    conn.commit()
                
                # 3. Friday Auto-Exit (Prop Firm Safety)
                now_utc = datetime.now(timezone.utc)
                # Friday (4) after 16:00 EST (approx 21:00 UTC)
                if now_utc.weekday() == 4 and now_utc.hour >= 20: 
                    logger.info("🕒 Friday 4 PM EST Detected. Closing all open risk for Weekend Mode.")
                    positions = mt5.positions_get()
                    if positions:
                        for p in positions:
                            logger.info(f"💾 Fast-Closing Position: {p.symbol} (Ticket: {p.ticket})")
                            # Simple Market Close
                            close_request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": p.symbol,
                                "volume": p.volume,
                                "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": p.ticket,
                                "price": mt5.symbol_info_tick(p.symbol).bid if p.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(p.symbol).ask,
                                "deviation": 20,
                                "magic": 999000,
                                "comment": "Apex Friday Exit",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_FOK,
                            }
                            # Handle filling
                            s_info = mt5.symbol_info(p.symbol)
                            if (s_info.filling_mode & 2) != 0: close_request["type_filling"] = mt5.ORDER_FILLING_IOC
                            
                            res = mt5.order_send(close_request)
                            if res.retcode != mt5.TRADE_RETCODE_DONE:
                                logger.error(f"Failed to close Friday position {p.ticket}: {res.comment}")
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Apex Connect Stopped.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main_loop()
