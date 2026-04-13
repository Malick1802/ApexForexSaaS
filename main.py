# =============================================================================
# ApexForex SaaS - The Executive
# =============================================================================
"""
The 'Executive' script that manages the autonomous lifecycle of the SaaS.

Responsibilities:
1. Hourly Loop: Wakes up every hour.
2. Data Fetch: Updates market data using DataEngine.
3. Prediction: Runs Specialist Models via InferenceEngine.
4. Dashboard Update: (Auto-handled via DB update).
5. Alerts: Sends High-Precision (>88%) Telegram signals.
"""

import time
import logging
import schedule
import os
import psutil
from datetime import datetime
from core.core.inference import InferenceEngine
from core.core.notifications import NotificationManager
from core.core.guardrail import get_guardrail

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("executive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Executive")

def job(engine, notifier):
    """The High-Frequency Job."""
    # ── 0. Safety Guardrail (Prop Firm Compliance) ─────────────────
    guard = get_guardrail()
    status = guard.get_safety_status()
    
    if not status['safe']:
        logger.warning(f"🛑 SAFETY HALT: {status['reason']} (Drawdown: {status['drawdown']:.1f}%)")
        return # Skip this cycle

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"⏰ Tick! Starting High-Frequency Cycle (Memory: {mem_mb:.1f} MB)...")
    
    try:
        # ── 1. Global Notification Poller (PRIORITY) ───────────────────────
        # Handle NEW signals immediately (from Scan, Dashboard, or Manual)
        all_new = [s for s in engine.db.get_recent_signals(limit=20) if s.get('status') == 'NEW']
        if all_new:
            logger.info(f"🔍 Poller found {len(all_new)} NEW signals to actuate.")
            for res in all_new:
                symbol = res['symbol']
                signal = res['signal']
                conf = res['confidence'] or 0.0
                regime = res.get('regime', 'UNKNOWN')
                target = res.get('regime_threshold') or 0.70
                
                is_valid_entry = conf >= target and signal in ["BUY", "SELL"]
                
                if is_valid_entry:
                    logger.info(f"🔥 ACTUATING: {symbol} {signal} ({conf:.1%})")
                    sent = notifier.send_signal_alert(res)
                    if sent:
                        engine.db.update_signal_status(res['id'], 'SENT')
                elif signal != "WAIT":
                    logger.info(f"👀 WATCH: {symbol} {signal} ({conf:.1%} | {regime}) (Hurdle: {target:.0%})")
                    # Mark as WATCH so we don't spam the log every 5 mins with the same setup
                    engine.db.update_signal_status(res['id'], 'WATCH')

        # ── 2. Precise Signal Resolution ──────────────────────────────
        active_signals = engine.db.get_active_signals()
        if active_signals:
            logger.info(f"⚖️ Resolving {len(active_signals)} active trades...")
            price_map = {}
            # Group by symbol to fetch each only once
            symbols_to_check = list(set([s['symbol'] for s in active_signals]))
            
            for symbol in symbols_to_check:
                try:
                    # Fetch 5m candles (2 days for buffer)
                    df = engine.data_engine.fetch(symbol, interval="5m", days=2)
                    if df is not None and not df.empty:
                        # Find the oldest active signal for this symbol to slice the history
                        oldest_sig = min([s for s in active_signals if s['symbol'] == symbol], key=lambda x: x['timestamp'])
                        sig_time = datetime.fromisoformat(oldest_sig['timestamp'])
                        
                        history = df[df.index >= sig_time]
                        if not history.empty:
                            price_map[symbol] = {
                                'high': float(history['high'].max()),
                                'low': float(history['low'].min()),
                                'close': float(history['close'].iloc[-1])
                            }
                except Exception as e:
                    logger.error(f"Resolution fetch failed for {symbol}: {e}")
            
            resolved = engine.db.resolve_signals(price_map)
            if resolved:
                for s, r in resolved.items():
                    logger.info(f"🏁 RESOLVED: {s} -> {r}")

        # ── 3. Run New Inference Scan (Background) ──────────────────────
        logger.info("🧠 Starting comprehensive 31-pair market scan...")
        engine.run_all(win_rate="60%")
        
    except Exception as e:
        logger.error(f"❌ Cycle Failed: {e}", exc_info=True)

    logger.info("Cycle Complete. Sleeping...")

def main():
    logger.info("🚀 ApexForex Executive Starting...")
    
    # Global Persistence Layer (Loads models once at startup)
    logger.info("🧠 Loading Expert Model Fleet (31 pairs)... Please wait (~4 mins).")
    engine = InferenceEngine()
    notifier = NotificationManager()
    
    # Run once immediately for startup test
    job(engine, notifier)
    
    # Schedule for every 5 minutes
    schedule.every(5).minutes.do(job, engine, notifier)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
