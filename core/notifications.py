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
        self.alert_threshold = self.telegram_config.get('alert_threshold', 0.60) # Default to 60% for shadow visibility
        self.notify_shadow = self.telegram_config.get('notify_shadow_trades', True)

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

        # Priority: 1. Signal's specific regime hurdle, 2. Config threshold, 3. Default 70%
        alert_threshold = signal_data.get('regime_threshold') or self.telegram_config.get('alert_threshold', 0.70)
        
        confidence = signal_data.get('confidence', 0)
        symbol = signal_data.get('symbol', 'UNKNOWN')
        signal = signal_data.get('signal', 'WAIT')
        is_shadow = signal_data.get('is_shadow_alert', False)

        if is_shadow and not self.notify_shadow:
            return False

        if confidence < alert_threshold:
            logger.info(f"Telegram alert skipped for {symbol} {signal} (Confidence {confidence*100:.1f}% < {alert_threshold*100:.1f}%)")
            return False

        label = "SHADOW" if is_shadow else "CERTIFIED"
        logger.info(f"Sending Telegram alert [{label}] for {symbol} {signal} (Confidence {confidence*100:.1f}%)")
            
        # Icon based on signal
        icon = "⚪"
        if signal_data['signal'] == "BUY": icon = "🟢"
        if signal_data['signal'] == "SELL": icon = "🔴"
        
        # ── 3. Format Message ──
        # Using Markdown v1 for maximum compatibility across TG clients
        tp_price = signal_data.get('tp_price', 0.0)
        sl_price = signal_data.get('sl_price', 0.0)
        tp_pips = signal_data.get('tp_pips', 0)
        sl_pips = signal_data.get('sl_pips', 0)
        
        tp_str = f"🎯 TP: `{tp_price:.5f}` (`+{tp_pips}p`)" if tp_price else "🎯 TP: `N/A`"
        sl_str = f"🛑 SL: `{sl_price:.5f}` (`-{sl_pips}p`)" if sl_price else "🛑 SL: `N/A`"
        
        # Volume/Model metadata label
        volume = signal_data.get('model_trades', 0)
        vol_label = f"`{volume} Trades`" if volume > 0 else "`Standard Pool`"
        
        msg = (
            f"{icon} *{signal_data['signal']} {signal_data['symbol']}*\n"
            f"Precision: `{signal_data['confidence']:.1%}`\n"
            f"📊 *Risk: 0.5% | Lots: {signal_data.get('suggested_lots', 0.01)}*\n"
            f"Entry: `{signal_data['price_at_signal']:.5f}`\n"
            f"{tp_str}\n"
            f"{sl_str}\n"
            f"Time: `{datetime.fromisoformat(signal_data['timestamp']).strftime('%H:%M UTC')}`"
        )
        
        if signal_data.get('is_shadow_alert'):
            msg = f"👻 *SHADOW / PAPER TRADE* 👻\n" + msg
            
        return self.send_telegram_message(msg)
