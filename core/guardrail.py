import logging
import sqlite3
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def detect_account_tier(balance: float) -> float:
    """
    Detect standard prop firm challenge tier (e.g. 5k, 10k, 25k, 50k, 100k, 200k, 500k, 1M).
    If balance is within 15% of a standard tier, returns the tier baseline so drawdown calculations
    strictly conform to prop firm rules (which evaluate drawdown against initial challenge balance).
    """
    if balance <= 0:
        return 10000.0
    tiers = [5000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000]
    for t in tiers:
        if abs(balance - t) / t <= 0.15:
            return float(t)
    return float(balance)

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

    def get_safety_status(self, exec_engine = None) -> Dict[str, Any]:
        """
        Evaluate if it is safe to generate new signals.
        Enforces:
          1. Weekend mode / Friday 14:00 UTC entry cutoff.
          2. Dynamic Daily drawdown kill switch: Scales with account tier
             (e.g. 4.5% = $450 on $10k, $4,500 on $100k, $45,000 on $1M).
             Closes all open MT5 positions immediately, resolves DB signals, and halts trading for the day.
        Returns: { 'safe': bool, 'reason': str, 'drawdown': float }
        """
        conf = self.config.get('safety', {})
        if not conf.get('enabled', True):
            return {'safe': True, 'reason': 'PROG_DISABLED', 'drawdown': 0.0}

        # 1. Check Weekend Mode (Friday 14:00 UTC cutoff & Sunday open buffer)
        is_weekend, weekend_reason = self._is_weekend_mode(conf)
        if is_weekend:
            return {'safe': False, 'reason': weekend_reason, 'drawdown': 0.0}

        # 2. Check Daily Drawdown & Emergency Kill Switch
        state = self._load_state()
        if state.get("halted_for_day", False):
            halt_dd = state.get("halt_drawdown", 450.0)
            return {
                'safe': False,
                'reason': f"DAILY_DRAWDOWN_KILLSWITCH_ACTIVE: Trading halted for today (Daily loss reached ${halt_dd:.2f}). Resets at 00:00 CEST.",
                'drawdown': 100.0
            }

        result = self._calculate_daily_drawdown()
        drawdown = result["drawdown_pct"]
        dollar_drawdown = result.get("dollar_drawdown", 0.0)
        max_loss_amount = result.get("max_loss_amount", 450.0)
        account_tier = result.get("account_tier", 10000.0)
        max_dd_pct = float(conf.get('max_daily_drawdown_pct', 4.5))

        # Trigger Kill Switch if dollar loss or percentage floor breached
        if dollar_drawdown >= max_loss_amount or drawdown >= max_dd_pct:
            self.execute_drawdown_killswitch(dollar_drawdown, max_loss_amount, exec_engine)
            return {
                'safe': False,
                'reason': f"DAILY_DRAWDOWN_KILLSWITCH_TRIGGERED: Daily loss reached ${dollar_drawdown:.2f} (Limit: ${max_loss_amount:.2f} [{max_dd_pct:.1f}% of ${account_tier:,.0f} tier]). All positions liquidated.",
                'drawdown': drawdown
            }

        return {'safe': True, 'reason': 'OK', 'drawdown': drawdown}

    def execute_drawdown_killswitch(self, dollar_drawdown: float, max_loss_amount: float, exec_engine = None):
        """
        Emergency Drawdown Kill Switch:
        1. Closes ALL open MT5 positions immediately.
        2. Resolves active signals in signals.db.
        3. Sets halted_for_day = True until midnight rollover.
        4. Sends high-priority Telegram emergency alert.
        """
        logger.critical(f"🚨 EMERGENCY KILL SWITCH: Daily loss ${dollar_drawdown:.2f} reached limit ${max_loss_amount:.2f}!")

        # 1. Update state to halt trading for today
        state = self._load_state()
        state["halted_for_day"] = True
        state["halt_time"] = datetime.now(timezone.utc).isoformat()
        state["halt_drawdown"] = dollar_drawdown
        self._save_state(state)

        # 2. Close all open positions in MT5
        closed_positions = []
        try:
            from core.mt5_connector import get_mt5
            mt5 = get_mt5()
            if mt5:
                positions = mt5.positions_get()
                if positions:
                    logger.critical(f"🛑 Kill Switch: Closing {len(positions)} open MT5 positions immediately!")
                    for p in positions:
                        tick = mt5.symbol_info_tick(p.symbol)
                        if not tick:
                            continue
                        close_price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
                        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

                        filling = mt5.ORDER_FILLING_FOK
                        s_info = mt5.symbol_info(p.symbol)
                        if s_info:
                            if (s_info.filling_mode & 1):
                                filling = mt5.ORDER_FILLING_FOK
                            elif (s_info.filling_mode & 2):
                                filling = mt5.ORDER_FILLING_IOC
                            else:
                                filling = mt5.ORDER_FILLING_RETURN

                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": p.symbol,
                            "volume": p.volume,
                            "type": close_type,
                            "position": p.ticket,
                            "price": close_price,
                            "deviation": 25,
                            "magic": 999450,
                            "comment": "Apex $450 DD Stop",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": filling,
                        }
                        res = mt5.order_send(close_req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"✅ Emergency Close: Position {p.symbol} (Ticket #{p.ticket}) closed at {close_price}")
                            closed_positions.append(f"{p.symbol} (#{p.ticket})")
                        else:
                            comment = res.comment if res else "No response"
                            logger.error(f"❌ Failed emergency close for {p.symbol} ticket {p.ticket}: {comment}")
        except Exception as e:
            logger.error(f"Error executing MT5 emergency liquidation: {e}", exc_info=True)

        # 2b. Broadcast emergency liquidation to all copy trading accounts
        try:
            from scripts.multi_executor import close_all_positions_for_all_users
            close_all_positions_for_all_users(reason="Daily Drawdown Kill Switch")
        except Exception as _gke:
            logger.error(f"Failed to close secondary accounts on drawdown killswitch: {_gke}")

        # 3. Resolve active signals in signals.db
        try:
            from core.database import SignalDatabase
            db = SignalDatabase()
            active_signals = db.get_active_signals(include_hidden=True)
            if active_signals:
                for sig in active_signals:
                    sig_id = sig['id']
                    sym = sig['symbol']
                    cur_price = sig.get('price_at_signal')
                    try:
                        from core.mt5_connector import get_mt5
                        m = get_mt5()
                        if m:
                            t = m.symbol_info_tick(sym)
                            if t:
                                cur_price = t.bid if sig['signal'] == 'BUY' else t.ask
                    except Exception:
                        pass
                    db.update_signal_outcome(
                        sig_id,
                        'FAIL',
                        exit_price=cur_price,
                        exit_reason=f"Emergency Kill Switch: Daily Drawdown Limit Reached (${max_loss_amount:.0f})"
                    )
                logger.info(f"Resolved {len(active_signals)} active signals in DB with Kill Switch exit reason.")
        except Exception as e:
            logger.error(f"Error resolving active signals during killswitch: {e}", exc_info=True)

        # 4. Dispatch Telegram emergency notification
        try:
            from core.notifications import NotificationManager
            notifier = NotificationManager()
            alert_msg = (
                f"🚨 *EMERGENCY DRAWDOWN KILL SWITCH ACTIVATED* 🚨\n\n"
                f"• *Daily Drawdown:* `${dollar_drawdown:.2f}` (Limit: `${max_loss_amount:.2f}`)\n"
                f"• *Positions Liquidated:* {len(closed_positions)} open trades closed\n"
                f"• *Status:* Trading *HALTED* for the rest of the day.\n"
                f"• *Automatic Reset:* 00:00 CEST (22:00 UTC rollover)."
            )
            notifier.send_telegram_message(alert_msg)
        except Exception as e:
            logger.error(f"Failed to send emergency Telegram notification: {e}")

    def _is_weekend_mode(self, conf: Dict) -> (bool, str):
        """Check if trading should be halted for the weekend using dynamic NY market hours."""
        from core.market_hours import is_weekend_halt
        return is_weekend_halt()

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
            max_dd_pct = float(conf.get('max_daily_drawdown_pct', 4.5))

            # --- Midnight snapshot: store balance at the start of each CEST day ---
            # Also reset snapshot if the connected account login ID has changed (e.g. migration)
            account_login = int(account.login)
            if (
                state.get("date") != trading_day
                or state.get("midnight_balance", 0.0) == 0.0
                or state.get("account_login") != account_login
            ):
                # First time seeing this trading day or account changed — snapshot the balance
                midnight_balance = current_balance

                # Detect and persist initial balance tier (e.g. $10k, $100k, $1M)
                tier = detect_account_tier(current_balance)
                initial_balance = tier
                logger.info(f"Initial Balance Tier Locked for #{account_login}: ${initial_balance:,.2f}")

                state = {
                    "date": trading_day,
                    "account_login": account_login,
                    "midnight_balance": midnight_balance,
                    "initial_balance": initial_balance,
                    "account_tier": tier,
                    "halted_for_day": False
                }
                self._save_state(state)
                logger.info(
                    f"[FTMO Guardrail] Account #{account_login} Active | Date: {trading_day} "
                    f"| Midnight Balance: ${midnight_balance:,.2f} "
                    f"| Account Tier: ${tier:,.2f} | Kill Switch Reset."
                )
            else:
                midnight_balance = state["midnight_balance"]
                initial_balance = state.get("initial_balance", current_balance)
                tier = state.get("account_tier") or detect_account_tier(initial_balance)

            # --- Dynamic Daily Loss Cap Formula ---
            # Scales dynamically with account tier/initial balance:
            # $10,000 account -> $450 (4.5%)
            # $100,000 account -> $4,500 (4.5%)
            # $1,000,000 account -> $45,000 (4.5%)
            if 'max_daily_drawdown_amount' in conf and conf['max_daily_drawdown_amount'] is not None:
                max_loss_amount = float(conf['max_daily_drawdown_amount'])
            else:
                max_loss_amount = (max_dd_pct / 100.0) * tier
            daily_floor = midnight_balance - max_loss_amount

            # Drawdown = how far equity has fallen below midnight balance
            # (negative means we are in profit, clamped to 0)
            dollar_drawdown = midnight_balance - current_equity
            if dollar_drawdown < 0:
                dollar_drawdown = 0.0

            drawdown_pct = (dollar_drawdown / tier) * 100.0

            logger.debug(
                f"[FTMO Guardrail] Equity: ${current_equity:,.2f} | "
                f"Floor: ${daily_floor:,.2f} | Drawdown: ${dollar_drawdown:,.2f} ({drawdown_pct:.2f}% of ${tier:,.2f})"
            )

            return {
                "drawdown_pct": drawdown_pct,
                "dollar_drawdown": dollar_drawdown,
                "equity": current_equity,
                "floor": daily_floor,
                "midnight_balance": midnight_balance,
                "max_loss_amount": max_loss_amount,
                "account_tier": tier
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
