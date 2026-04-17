import sys
import os
import time
import logging
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# Fix module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mt5_connector import get_mt5

# Configure Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SENTINEL - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "sentinel.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Sentinel")

DB_PATH = Path("signals.db").absolute()

class LightSentinel:
    def __init__(self):
        self.db_path = DB_PATH
        self.mt5_initialized = False
        logger.info(f"Sentinel initialized. DB: {self.db_path}")

    def init_mt5(self):
        self.mt5 = get_mt5()
        if self.mt5 is None:
            logger.error("MT5 Bridge Connection Failed")
            return False
        self.mt5_initialized = True
        return True

    def get_active_signals(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            # NEW: Monitor ALL active outcomes, including 'WAIT' shadow trades
            query = """
            SELECT id, symbol, signal, expert_signal, price_at_signal, tp_price, sl_price, mt5_ticket, timestamp 
            FROM signals 
            WHERE outcome = 'ACTIVE' 
            """
            cursor = conn.cursor()
            rows = cursor.execute(query).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"DB Read Error: {e}")
            return []

    def update_signal(self, signal_id, outcome, exit_price, exit_reason):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE signals SET outcome = ?, exit_price = ?, exit_reason = ? WHERE id = ?",
                (outcome, exit_price, exit_reason, signal_id)
            )
            conn.commit()
            conn.close()
            logger.info(f"✅ DB Updated: ID {signal_id} -> {outcome} ({exit_reason}) @ {exit_price}")
        except Exception as e:
            logger.error(f"DB Update Error: {e}")

    def check_market(self, signals):
        if not signals:
            return

        if not self.mt5_initialized and not self.init_mt5():
            return

        for sig in signals:
            sym = sig['symbol']
            sig_id = sig['id']
            # Use expert_signal for resolution, but if it was overridden to WAIT by safety layers, 
            # fall back to the raw `signal` so Shadow/Benched trades can be properly graded.
            direction = sig.get('expert_signal')
            if not direction or direction == "WAIT":
                direction = sig.get('signal')
            entry_price = sig['price_at_signal']
            tp_price = sig.get('tp_price')
            sl_price = sig.get('sl_price')

            if direction not in ('BUY', 'SELL'):
                continue

            # Get Real-Time MT5 Tick (Direct Price Update)
            tick = self.mt5.symbol_info_tick(sym)
            if not tick:
                logger.warning(f"Could not get tick for {sym}")
                continue

            current_price = tick.bid if direction == "BUY" else tick.ask
            
            if not tp_price or not sl_price:
                # logger.debug(f"Skipping {sym} (No TP/SL in DB)")
                continue

            outcome = None
            reason = ""
            
            if direction == "BUY":
                if tick.bid >= tp_price:
                    outcome = "SUCCESS"
                    reason = "TP Hit (Price)"
                elif tick.bid <= sl_price:
                    outcome = "FAIL"
                    reason = "SL Hit (Price)"
                    
            elif direction == "SELL":
                if tick.ask <= tp_price:
                    outcome = "SUCCESS"
                    reason = "TP Hit (Price)"
                elif tick.ask >= sl_price:
                    outcome = "FAIL"
                    reason = "SL Hit (Price)"

            if outcome:
                # Update Database result (Certification history)
                # Note: We NO LONGER send an explicit close order to MT5.
                # MT5 is expected to handle its own TP/SL closure natively.
                self.update_signal(sig_id, outcome, current_price, reason)
            else:
                # Periodic log for tracking
                logger.info(f"Monitoring {sym} {direction} @ {current_price:.5f} (TP: {tp_price:.5f}, SL: {sl_price:.5f})")

    def run(self):
        logger.info("Sentinel loop started (Price-Native Resolution)")
        while True:
            try:
                signals = self.get_active_signals()
                if signals:
                    self.check_market(signals)
                else:
                    logger.info("No active signals to monitor.")
            except Exception as e:
                logger.error(f"Sentinel Loop Interruption: {e}")
            
            time.sleep(10) # 10-second price polling

if __name__ == "__main__":
    import threading
    
    # 1. Start MT5 Bridge (ApexConnect) in background thread
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from apex_connect import main_loop as apex_main_loop
        
        mt5_thread = threading.Thread(target=apex_main_loop, daemon=True, name="ApexConnect-MT5")
        mt5_thread.start()
        logger.info("🔌 MT5 Auto-Execution Bridge started (background thread)")
    except Exception as e:
        logger.warning(f"⚠️ MT5 bridge failed: {e}")
    
    # 2. Start Sentinel monitoring
    sentinel = LightSentinel()
    sentinel.run()
