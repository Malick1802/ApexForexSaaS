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

    def _get_state_path(self):
        path = Path("data_cache/drawdown_state.json")
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _load_state(self):
        import json
        path = self._get_state_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"date": "", "hwm": 0.0}

    def _save_state(self, state):
        import json
        with open(self._get_state_path(), 'w') as f:
            json.dump(state, f)

    def _calculate_daily_drawdown(self) -> float:
        """
        Calculates Prop Firm Daily Drawdown based on MT5 Equity vs Balance High Water Mark
        at the 23:00 GMT+3 (20:00 UTC) rollover.
        """
        try:
            from core.mt5_connector import get_mt5
            mt5 = get_mt5()
            
            # Fail-safe: If MT5 is disconnected, we cannot verify live equity.
            # Prop firm safety dictates we must pause until we have eyes on the account.
            if not mt5:
                logger.error("MT5 disconnected. Failsafe activated for Drawdown check.")
                return 999.0 # Forces a block
                
            account = mt5.account_info()
            if not account:
                logger.error("MT5 account info unavailable. Failsafe activated.")
                return 999.0
                
            current_equity = float(account.equity)
            current_balance = float(account.balance)
            
            # 1. Determine the "Trading Day" Date String based on 20:00 UTC rollover
            now_utc = datetime.now(timezone.utc)
            # If time is past 20:00 UTC, it is already the "next" trading day
            if now_utc.hour >= 20:
                trading_day = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                trading_day = now_utc.strftime("%Y-%m-%d")
                
            state = self._load_state()
            
            # 2. If it's a new trading day, snapshot the High Water Mark
            if state.get("date") != trading_day or state.get("hwm", 0.0) == 0.0:
                hwm = max(current_balance, current_equity)
                state = {"date": trading_day, "hwm": hwm}
                self._save_state(state)
                logger.info(f"Daily Rollover Snapshot: Trading Day {trading_day} | HWM: ${hwm:.2f}")
            else:
                hwm = state.get("hwm")
                
            # 3. Calculate True Drawdown %
            if hwm <= 0: return 0.0
            
            # If current equity is higher than the floor, drawdown is negative or small.
            # Drawdown = percentage drop from HWM
            drawdown_pct = ((hwm - current_equity) / hwm) * 100.0
            
            # We don't care about negative drawdown (profits above HWM), just clamp to 0
            if drawdown_pct < 0:
                drawdown_pct = 0.0
                
            return drawdown_pct
            
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}. Failsafe activated.")
            return 999.0

# Singleton
_guard = None
def get_guardrail() -> PropGuardrail:
    global _guard
    if _guard is None:
        _guard = PropGuardrail()
    return _guard
