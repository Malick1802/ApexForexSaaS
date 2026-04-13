import logging
import sqlite3
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PropGuardrail:
    """
    Safety Guardrail for Prop Firm Compliance.
    Monitors daily drawdown and weekend risk.
    """
    
    def __init__(self, db_path: str = "signals.db", config_path: str = "config.yaml"):
        self.db_path = db_path
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Guardrail failed to load config: {e}")
            return {}

    def get_safety_status(self) -> Dict[str, Any]:
        """
        Evaluate if it is safe to generate new signals.
        Returns: { 'safe': bool, 'reason': str, 'drawdown': float }
        """
        conf = self.config.get('safety', {})
        if not conf.get('enabled', True):
            return {'safe': True, 'reason': 'PROG_DISABLED', 'drawdown': 0.0}

        # 1. Check Weekend Mode (Friday Exit)
        is_weekend, weekend_reason = self._is_weekend_mode(conf)
        if is_weekend:
            return {'safe': False, 'reason': weekend_reason, 'drawdown': 0.0}

        # 2. Check Daily Drawdown
        drawdown = self._calculate_daily_drawdown()
        max_dd = conf.get('max_daily_drawdown_pct', 2.0)
        
        if drawdown >= max_dd:
            return {
                'safe': False, 
                'reason': f"DAILY_DRAWDOWN_HIT ({drawdown:.2f}% >= {max_dd}%)",
                'drawdown': drawdown
            }

        return {'safe': True, 'reason': 'OK', 'drawdown': drawdown}

    def _is_weekend_mode(self, conf: Dict) -> (bool, str):
        """Check if trading should be halted for the weekend."""
        now_utc = datetime.now(timezone.utc)
        # Convert UTC to EST (approximate for Friday exit logic)
        now_est = now_utc - timedelta(hours=5)
        
        # Friday = 4
        if now_est.weekday() == 4:
            exit_hour = conf.get('close_friday_hour_est', 16)
            if now_est.hour >= exit_hour:
                return True, "WEEKEND_EXIT_ACTIVE"
        
        # Saturday (5) & Sunday (6)
        if now_est.weekday() in (5, 6):
            # DISABLED for testing:
            # return True, "MARKET_CLOSED"
            pass
            
        return False, ""

    def _calculate_daily_drawdown(self) -> float:
        """
        Calculate total loss for today as % of estimated account size.
        Uses signals.db to sum realized 'FAIL' outcomes today.
        """
        try:
            # We assume a base account size for % calculation if MT5 is not connected
            # In production, this would pull real equity from MT5
            base_equity = 100000.0 
            risk_pct = self.config.get('mt5', {}).get('risk_value', 0.5)
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get start of today (UTC)
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
            
            # Count Failed trades today
            cursor.execute("""
                SELECT COUNT(*) as failures 
                FROM signals 
                WHERE timestamp >= ? AND outcome = 'FAIL'
            """, (today_start,))
            
            row = cursor.fetchone()
            failures = row['failures'] if row else 0
            
            # Estimated Drawdown = failures * risk_per_trade
            # This is a robust proxy for drawdown when live equity is transient
            total_drawdown = failures * risk_pct
            
            conn.close()
            return total_drawdown
            
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}")
            return 0.0

# Singleton
_guard = None
def get_guardrail() -> PropGuardrail:
    global _guard
    if _guard is None:
        _guard = PropGuardrail()
    return _guard
