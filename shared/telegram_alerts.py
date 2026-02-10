# =============================================================================
# Telegram Alert System
# =============================================================================
"""
Handles sending high-priority trading alerts to Telegram.

CRITICAL RULE: Only sends alerts if confidence is > 88%.
This ensures we protect the high win-rate standard of the Specialist models.
"""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Simple wrapper for Telegram Bot API.
    """
    
    def __init__(self):
        # Load from environment or config
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not found. Alerts will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram Bot initialized.")

    def send_message(self, message: str) -> bool:
        """Send a raw text message."""
        if not self.enabled:
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_signal(
        self, 
        symbol: str, 
        signal: str, 
        confidence: float, 
        entry_price: float, 
        sl_price: float, 
        tp_price: float,
        timeframe: str = "1H"
    ) -> bool:
        """
        Send a formatted trading signal IF confidence > 88%.
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            signal: 'BUY' or 'SELL'
            confidence: Model probability (0.0 to 1.0)
            entry_price: Current price
            sl_price: Stop Loss price
            tp_price: Take Profit price
            timeframe: Chart timeframe
            
        Returns:
            True if sent, False if filtered or failed.
        """
        # CRITICAL FILTER: Protect the Win Rate
        if confidence <= 0.88:
            logger.info(f"Signal filtered (Confidence {confidence:.2%} <= 88%)")
            return False
            
        emoji = "🚀" if signal == "BUY" else "🔻"
        conf_str = f"{confidence:.1%}"
        
        # Calculate Risk/Reward
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        message = (
            f"🚨 <b>HIGH PRECISION ALERT</b> 🚨\n\n"
            f"<b>Pair:</b> {symbol} ({timeframe})\n"
            f"<b>Action:</b> {signal} {emoji}\n"
            f"<b>Confidence:</b> {conf_str}\n\n"
            f"🎯 <b>Entry:</b> {entry_price:.5f}\n"
            f"🛑 <b>SL:</b> {sl_price:.5f}\n"
            f"💰 <b>TP:</b> {tp_price:.5f}\n\n"
            f"<i>R:R Ratio: 1:{rr_ratio:.1f}</i>"
        )
        
        logger.info(f"Sending HIGH CONFIDENCE alert for {symbol} ({conf_str})")
        return self.send_message(message)

# Global instance
bot = TelegramBot()

def send_alert(symbol, signal, confidence, entry, sl, tp):
    """Convenience wrapper."""
    return bot.send_signal(symbol, signal, confidence, entry, sl, tp)
