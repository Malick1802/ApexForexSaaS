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
        result = self._calculate_daily_drawdown()
        drawdown = result["drawdown_pct"]
        max_dd = conf.get('max_daily_drawdown_pct', 2.5)

        if drawdown >= max_dd:
            floor = result.get('floor', 0)
            equity = result.get('equity', 0)
            return {
                'safe': False,
                'reason': f"DAILY_DRAWDOWN_HIT ({drawdown:.2f}% >= {max_dd}% | Equity: ${equity:.2f} vs Floor: ${floor:.2f})",
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
        return {"date": "", "midnight_balance": 0.0, "initial_balance": 0.0}

    def _save_state(self, state):
        import json
        with open(self._get_state_path(), 'w') as f:
            json.dump(state, f)

    def _calculate_daily_drawdown(self) -> dict:
        """
        Calculates FTMO-compliant Daily Drawdown.

        FTMO Rule:
          - Daily floor = (account balance at midnight CEST) - (max_daily_drawdown_pct% x initial_balance)
          - Block if current_equity (includes floating P&L) <= daily floor
          - Rollover: 00:00 CEST = 22:00 UTC (summer / CEST = UTC+2)
          - Loss amount is FIXED per day based on initial challenge balance.
        """
        try:
            from core.mt5_connector import get_mt5
            mt5 = get_mt5()

            if not mt5:
                logger.error("MT5 disconnected. Failsafe activated.")
                return {"drawdown_pct": 999.0, "equity": 0, "floor": 0, "midnight_balance": 0}

            account = mt5.account_info()
            if not account:
                logger.error("MT5 account info unavailable. Failsafe activated.")
                return {"drawdown_pct": 999.0, "equity": 0, "floor": 0, "midnight_balance": 0}

            current_equity = float(account.equity)   # includes open floating P&L
            current_balance = float(account.balance) # closed trades only

            # --- CEST Rollover: 00:00 CEST = 22:00 UTC (UTC+2 in summer) ---
            now_utc = datetime.now(timezone.utc)
            now_cest = now_utc + timedelta(hours=2)  # CEST = UTC+2
            trading_day = now_cest.strftime("%Y-%m-%d")

            state = self._load_state()
            conf = self.config.get('safety', {})
            max_dd_pct = conf.get('max_daily_drawdown_pct', 2.5)

            # --- Midnight snapshot: store balance at the start of each CEST day ---
            if state.get("date") != trading_day or state.get("midnight_balance", 0.0) == 0.0:
                # First time seeing this trading day — snapshot the balance
                midnight_balance = current_balance

                # Persist initial balance if this is the first ever run
                initial_balance = state.get("initial_balance", 0.0)
                if initial_balance <= 0.0:
                    initial_balance = current_balance
                    logger.info(f"Initial Balance Locked: ${initial_balance:.2f}")

                state = {
                    "date": trading_day,
                    "midnight_balance": midnight_balance,
                    "initial_balance": initial_balance
                }
                self._save_state(state)
                logger.info(
                    f"[FTMO Guardrail] Daily Rollover | Date: {trading_day} "
                    f"| Midnight Balance: ${midnight_balance:.2f} "
                    f"| Initial Balance: ${initial_balance:.2f}"
                )
            else:
                midnight_balance = state["midnight_balance"]
                initial_balance = state.get("initial_balance", midnight_balance)

            # --- FTMO Formula ---
            # max_loss_amount = fixed dollar amount per day (e.g. 2.5% of $10,000 = $250)
            max_loss_amount = (max_dd_pct / 100.0) * initial_balance
            daily_floor = midnight_balance - max_loss_amount

            # Drawdown = how far equity has fallen below midnight balance
            # (negative means we are in profit, clamped to 0)
            dollar_drawdown = midnight_balance - current_equity
            if dollar_drawdown < 0:
                dollar_drawdown = 0.0

            drawdown_pct = (dollar_drawdown / initial_balance) * 100.0

            logger.debug(
                f"[FTMO Guardrail] Equity: ${current_equity:.2f} | "
                f"Floor: ${daily_floor:.2f} | Drawdown: {drawdown_pct:.2f}%"
            )

            return {
                "drawdown_pct": drawdown_pct,
                "equity": current_equity,
                "floor": daily_floor,
                "midnight_balance": midnight_balance,
                "max_loss_amount": max_loss_amount
            }

        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}. Failsafe activated.")
            return {"drawdown_pct": 999.0, "equity": 0, "floor": 0, "midnight_balance": 0}

# Singleton
_guard = None
def get_guardrail() -> PropGuardrail:
    global _guard
    if _guard is None:
        _guard = PropGuardrail()
    return _guard
