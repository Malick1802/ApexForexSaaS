# =============================================================================
# Notification Manager
# =============================================================================
"""
Manager for sending external notifications (Telegram, Email, etc.).
"""

import logging
import requests
import yaml
from typing import Dict, Any, Optional
from datetime import datetime

from data_pipeline.engine import DataEngine

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Handles sending alerts to configured channels.
    """
    
    def __init__(self):
        self._load_config()
        
    def _load_config(self):
        """Load configuration using DataEngine's logic."""
        # reusing DataEngine to find/load config easily
        self.engine = DataEngine()
        self.config = self.engine.config
        
        self.telegram_config = self.config.get('notifications', {}).get('telegram', {})
        self.enabled = self.telegram_config.get('enabled', False)
        self.bot_token = self.telegram_config.get('bot_token', '')
        self.chat_id = self.telegram_config.get('chat_id', '')
        self.alert_threshold = self.telegram_config.get('alert_threshold', 0.85)

    def send_telegram_message(self, message: str) -> bool:
        """
        Send a raw message to Telegram.
        """
        if not self.enabled:
            logger.debug("Telegram alerts disabled.")
            return False
            
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram enabled but credentials missing.")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Telegram message sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_signal_alert(self, signal_data: Dict[str, Any]):
        """
        Format and send a trade signal alert.
        """
        if not self.enabled:
            return

        # Get threshold from config or default to 0.85
        alert_threshold = self.telegram_config.get('alert_threshold', 0.85)
        
        confidence = signal_data.get('confidence', 0)
        symbol = signal_data.get('symbol', 'UNKNOWN')
        signal = signal_data.get('signal', 'WAIT')

        if confidence < alert_threshold:
            logger.info(f"Telegram alert skipped for {symbol} {signal} (Confidence {confidence*100:.1f}% < {alert_threshold*100:.0f}%)")
            return

        logger.info(f"Sending Telegram alert for {symbol} {signal} (Confidence {confidence*100:.1f}%)")
            
        # Icon based on signal
        icon = "⚪"
        if signal_data['signal'] == "BUY": icon = "🟢"
        if signal_data['signal'] == "SELL": icon = "🔴"
        
        # The old threshold check is removed as it's replaced by the new logic above.
        # if signal['confidence'] < self.alert_threshold:
        #     # logger.info(f"Signal suppressed: Conf {signal['confidence']:.2%} < Threshold {self.alert_threshold:.2%}")
        #     return False

        # Format Message
        # 🟢 BUY EURUSD
        # Conf: 79.8% | Price: 1.1234
        # TP: 1.1284 | SL: 1.1209
        # ---------------------------
        
        tp_str = f"TP: `{signal_data.get('tp_price', 'N/A')}`"
        sl_str = f"SL: `{signal_data.get('sl_price', 'N/A')}`"
        
        msg = (
            f"{icon} *{signal_data['signal']} {signal_data['symbol']}*\n"
            f"Confidence: `{signal_data['confidence']:.1%}`\n"
            f"Entry: `{signal_data['price_at_signal']}`\n"
            f"🎯 {tp_str}\n"
            f"🛑 {sl_str}\n"
            f"Time: `{datetime.fromisoformat(signal_data['timestamp']).strftime('%H:%M UTC')}`"
        )
        
        return self.send_telegram_message(msg)
