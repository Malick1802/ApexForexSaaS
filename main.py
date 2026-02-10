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
from datetime import datetime
from core.inference import InferenceEngine
from shared.telegram_alerts import send_alert

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

def job():
    """The Hourly Job."""
    logger.info("⏰ Tick! Starting Hourly Cycle...")
    
    try:
        # Initialize Engine (Reloads models in case of updates)
        engine = InferenceEngine()
        
        # Run Inference on All Pairs
        results = engine.run_all()
        logger.info(f"Generated {len(results)} predictions.")
        
        # Process Alerts
        for res in results:
            symbol = res['symbol']
            signal = res['signal']
            conf = res['confidence']
            status = res.get('status', 'NEW')
            
            # Log result
            logger.info(f"{symbol}: {signal} ({conf:.1%}) - {status}")
            
            # Telegram Alert Logic (Strict > 88% handled by sender)
            if status == "NEW" and signal in ["BUY", "SELL"]:
                sent = send_alert(
                    symbol=symbol,
                    signal=signal,
                    confidence=conf,
                    entry_price=res['price_at_signal'],
                    sl_price=res['sl_price'],
                    tp_price=res['tp_price']
                )
                if sent:
                    logger.info(f"🚀 Alert sent for {symbol}!")
                    
    except Exception as e:
        logger.error(f"❌ Cycle Failed: {e}", exc_info=True)

    logger.info("Cycle Complete. Sleeping...")

def main():
    logger.info("🚀 ApexForex Executive Started.")
    logger.info("Waiting for next hour mark...")
    
    # Run once immediately for startup test
    job()
    
    # Schedule for every hour at minute 01
    schedule.every().hour.at(":01").do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
