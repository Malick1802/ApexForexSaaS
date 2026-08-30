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
        self.channel_id = self.telegram_config.get('channel_id', '')
        self.alert_threshold = self.telegram_config.get('alert_threshold', 0.52) # Default to 52% while Platt Calibrator trains
        self.notify_shadow = self.telegram_config.get('notify_shadow_trades', True)

    def send_telegram_message(self, message: str) -> bool:
        """
        Send a message to Telegram (Chat and Channel if configured).
        """
        if not self.enabled or not self.bot_token:
            return False
            
        targets = []
        if self.chat_id: targets.append(self.chat_id)
        if self.channel_id: targets.append(self.channel_id)
        
        if not targets:
            logger.warning("Telegram enabled but no targets (chat_id/channel_id) configured.")
            return False
            
        success = False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        for target in targets:
            payload = {
                "chat_id": target,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            try:
                response = requests.post(url, json=payload, timeout=15)
                response.raise_for_status()
                logger.info(f"Telegram message sent to {target}")
                success = True # At least one succeeded
            except Exception as e:
                logger.error(f"Failed to send Telegram message to {target}: {e}")
                
        return success

    def send_signal_alert(self, signal_data: Dict[str, Any]):
        """
        Format and send a trade signal alert.
        """
        if not self.enabled:
            return

        confidence = signal_data.get('confidence', 0)
        symbol = signal_data.get('symbol', 'UNKNOWN')
        signal = signal_data.get('signal', 'WAIT')
        is_shadow = signal_data.get('is_shadow_alert', False)

        # For Shadow Alerts, use the global config floor to allow visibility while benched.
        # For Certified alerts, respect the dynamic regime hurdle (0.65+ for trending etc.)
        if is_shadow:
            alert_threshold = self.telegram_config.get('alert_threshold', 0.52)
        else:
            alert_threshold = signal_data.get('regime_threshold') or self.telegram_config.get('alert_threshold', 0.52)

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
        
        # Format dynamically using the live config risk
        try:
            mt5_cfg = self.config.get('mt5', {})
            risk_type = mt5_cfg.get('risk_type', 'percent')
            risk_value = mt5_cfg.get('risk_value', 0.5)
            if risk_type == 'fixed_usd':
                risk_label = f"${risk_value:.2f} fixed"
            elif risk_type == 'fixed':
                risk_label = f"{risk_value} lots fixed"
            else:
                risk_label = f"{risk_value:.2f}%"
        except Exception:
            risk_label = "N/A"

        msg = (
            f"{icon} *{signal_data['signal']} {signal_data['symbol']}*\n"
            f"Precision: `{signal_data['confidence']:.1%}`\n"
            f"📊 *Risk: {risk_label} | Lots: {signal_data.get('suggested_lots', 0.01)}*\n"
            f"Entry: `{signal_data['price_at_signal']:.5f}`\n"
            f"{tp_str}\n"
            f"{sl_str}\n"
            f"Time: `{datetime.fromisoformat(signal_data['timestamp']).strftime('%H:%M UTC')}`"
        )
        
        if signal_data.get('is_shadow_alert'):
            msg = f"👻 *SHADOW / PAPER TRADE* 👻\n" + msg
            
        return self.send_telegram_message(msg)

    def send_periodic_performance_report(
        self,
        period: str = "both",
        risk_per_trade: float = 50.0,
        mode: str = "production",
        start_date: Optional[str] = None
    ) -> bool:
        """
        Generate and dispatch comprehensive weekly/monthly performance scorecard to Telegram.
        """
        try:
            from core.performance_report import PerformanceReporter
            reporter = PerformanceReporter()
            scorecard = reporter.generate_telegram_scorecard(period=period, risk_per_trade=risk_per_trade, start_date=start_date)
            return self.send_telegram_message(scorecard)
        except Exception as e:
            logger.error(f"Failed to generate or send periodic performance report: {e}")
            return False


