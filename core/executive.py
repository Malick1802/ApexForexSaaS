# =============================================================================
# Executive - Production Background Worker
# =============================================================================
"""
Autonomous signal generation engine that:
- Polls MT5/yfinance every 15 minutes
- Uses specialist models for prediction
- Sends Telegram alerts for high-confidence signals (>85%)
- Logs all activity to system.log
"""

import os
import sys
import logging
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from queue import Queue
from threading import Thread, Lock

import numpy as np
import pandas as pd
import yaml
import re
import requests
# Removed python-telegram-bot imports due to imghdr dependency crash in Python 3.13

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from core.database import SignalDatabase
from core.inference import InferenceEngine
from core.notifications import NotificationManager
try:
    from tensorflow import keras
    _KERAS_AVAILABLE = True
except Exception as _keras_err:
    keras = None
    _KERAS_AVAILABLE = False
import joblib


# =============================================================================
# Setup Logging
# =============================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "system_v2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# Rate limiting is handled centrally by providers


# =============================================================================
# =============================================================================
# Executive Engine
# =============================================================================

class ExecutiveEngine:
    """
    Production background worker for autonomous signal generation.
    
    Features:
    - 15-minute scanning interval
    - Rate limiting (via InferenceEngine's DataEngine)
    - High-Confidence "Apex" signals (Default)
    - Telegram alerts
    - Full activity logging
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        target_win_rate: str = "61%",  # Optimal institutional floor
        scan_interval_minutes: int = 1
    ):
        logger.info("="*70)
        logger.info("EXECUTIVE ENGINE - STARTING")
        logger.info("="*70)
        
        self.target_win_rate = target_win_rate
        self.scan_interval_minutes = scan_interval_minutes
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        # cooldown_minutes=0 so we can scan every interval without internal blocking
        self.inference_engine = InferenceEngine(config_path=config_path, cooldown_minutes=0)
        self.db = SignalDatabase()
        self.notifier = NotificationManager()
        
        # MT5 Trade Engine
        self.risk_pct = 0.005 # GetLeveraged.com compliance: 0.5% risk per trade
        
        # Recent signals tracker (deduplication)
        self._recent_signals: Dict[str, datetime] = {}
        self._cooldown_minutes = 240  # 4-hour cooldown to prevent repetitive 'Machine Gun' alerts
        
        # Loss cooldown tracker (anti-revenge trading)
        # Maps symbol -> datetime when the last SL hit occurred
        self._loss_cooldowns: Dict[str, datetime] = {}
        self._loss_cooldown_minutes = 240  # 4 hours
        
        # Blocked symbols (temporarily disabled due to poor win rate)
        self._blocked_symbols = set(self.config.get('blocked_symbols', []))
        if self._blocked_symbols:
            logger.info(f"🚫 BLOCKED SYMBOLS (disabled): {sorted(self._blocked_symbols)}")
        self.last_bayesian_update = datetime.now(timezone.utc).date()
        
        logger.info(f"Target Win Rate: {target_win_rate}")
        logger.info(f"Scan Interval: {scan_interval_minutes} minutes")
        logger.info(f"Telegram Alerts: {'Enabled' if self.notifier.enabled else 'Disabled'}")

        # ── STARTUP GATE RECOMPUTE ─────────────────────────────────────────────
        # Critical: recompute the performance gate on every startup before scanning.
        # Rationale: If the executive was offline (restart, crash, maintenance), any
        # signals that resolved via the sentinel or MT5 while we were down will NOT
        # have triggered the in-process recompute. The whitelist.json on disk could be
        # stale (e.g., showing APPROVED for a pair that later hit its SL). Without this,
        # the first scan after a restart reads the outdated gate and can authorize
        # a live signal for a BENCHED pair — exactly the EURUSD #968 incident.
        try:
            logger.info("Startup: Recomputing performance gate from live DB...")
            from core.performance_gate import PerformanceGate
            startup_gate = PerformanceGate()
            startup_gate.recompute_from_db(lookback_days=14)
            startup_gate.save_whitelist()
            logger.info("Startup: Performance gate synchronized. Ready to scan.")
        except Exception as _gate_err:
            logger.error(f"Startup gate recompute failed (non-fatal): {_gate_err}")

        logger.info("="*70)

    @property
    def mt5(self):
        """Dynamic connection to MT5 (prevents stale connection issues)."""
        from core.mt5_connector import get_mt5
        return get_mt5()
        
    def get_all_pairs(self) -> List[str]:
        """Proxy for DataEngine.get_all_pairs()."""
        return self.inference_engine.data_engine.get_all_pairs()

    def _run_daily_maintenance(self):
        """Automatically recompute Bayesian matrix when day rolls over."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_bayesian_update:
            logger.info("Date rolled over. Running Daily Bayesian Matrix Update...")
            try:
                from core.performance_gate import PerformanceGate
                gate = PerformanceGate()
                gate.recompute_from_db(lookback_days=14)
                gate.save_whitelist()
                self.last_bayesian_update = current_date
                logger.info("Daily Bayesian Matrix Update Complete (Rolling 14-day window maintained).")
            except Exception as e:
                logger.error(f"Matrix Update Failed: {e}", exc_info=True)

    def _calculate_lot_size(self, symbol: str, sl_pips: float) -> float:
        """
        Calculate lot size for 0.5% dynamic account risk.
        Logic: Lots = (Balance * 0.005) / (SL_Pips * Pip_Value_per_Lot)
        """
        try:
            if not self.mt5:
                return 0.01
                
            account = self.mt5.account_info()
            if not account:
                return 0.01
                
            balance = account.balance
            risk_usd = balance * self.risk_pct # 0.5%
            
            # Get pip value from DataEngine
            pip_value = self.inference_engine.data_engine.get_pip_value(symbol)
            
            # For most pairs, 1 standard lot = 100,000 units.
            # Pip value returned by DataEngine is 0.0001 or 0.01.
            # Standard calculation: Risk / (SL_distance * Pip_value_for_one_lot)
            # A 1.0 unit change in EURUSD (e.g. 1.05 -> 1.06) is 100 pips.
            # Pip_Value_per_Lot for MT5 is usually in Account Currency.
            
            symbol_info = self.mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
                
            # Use MT5's trade_tick_value (value of 1 pip for 1 lot in balance currency)
            # Note: tick_value is for 1 lot per tick.
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            
            # SL distance in price units
            sl_dist_price = sl_pips * pip_value
            
            if sl_dist_price <= 0:
                return 0.01
                
            # Lots = Risk_USD / (Loss_per_lot)
            loss_per_lot = (sl_dist_price / tick_size) * tick_value
            risk_lots = risk_usd / loss_per_lot
            
            # ── 1:30 PROP-FIRM LEVERAGE CONSTRAINT (GETLEVERAGED) ──
            # Physical limit of buying power. 
            # E.g. $10k account at 1:30 max position is ~$300k notional (approx 3.0 lots total)
            max_leverage = 30
            # Calculate the explicit margin required for 1.0 standard lot using MT5 api
            margin_per_lot = self.mt5.order_calc_margin(self.mt5.ORDER_TYPE_BUY, symbol, 1.0, symbol_info.ask)
            
            if margin_per_lot:
                # We use 90% of available margin to ensure we don't hit auto-liquidation walls
                max_margin_lots = (balance * 0.9) / margin_per_lot
            else:
                # Fallback purely derived from notional value
                notional_value_per_lot = symbol_info.ask * symbol_info.trade_contract_size
                max_margin_lots = ((balance * 0.9) * max_leverage) / notional_value_per_lot

            # Take the smallest between our Risk Profile and our Margin Reality
            lots = min(risk_lots, max_margin_lots)
            
            # Normalize strictly to broker step requirements
            step = symbol_info.volume_step
            lots = round(lots / step) * step
            lots = max(lots, symbol_info.volume_min)
            lots = min(lots, symbol_info.volume_max)
            
            logger.info(f"Risk Calc [{symbol}]: Bal=${balance:.2f}, Risk=${risk_usd:.2f} (0.5%), SL={sl_pips:.1f}p -> Margin Capped Lots={lots:.2f}")
            return lots
            
        except Exception as e:
            logger.error(f"Failed lot calculation for {symbol}: {e}")
            return 0.01

    def _check_drawdown_limits(self) -> bool:
        """
        Check if account is near daily loss limit, max trailing loss limit, or max open trades limit. 
        Returns True if safe to trade, False if blocked.
        """
        try:
            if not self.mt5:
                return True
                
            account = self.mt5.account_info()
            if not account:
                return True
                
            # 0. Max Open Trades Check
            max_open = self.config.get('mt5', {}).get('max_open_trades', 5)
            if max_open > 0:
                open_positions = self.mt5.positions_total()
                if open_positions >= max_open:
                    logger.critical(f"🛑 RISK SHIELD: Maximum open trades reached ({open_positions}/{max_open}). Blocking all Live Trades!")
                    return False
                
            now_utc = datetime.now(timezone.utc)
            # Use UTC midnight as the safest anchor for daily drawdown start 
            midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            
            deals = self.mt5.history_deals_get(midnight_utc, now_utc + timedelta(days=1))
            
            realized_profit_today = 0.0
            if deals:
                realized_profit_today = sum(d.profit + d.swap + d.commission for d in deals)
                
            # Active Daily PnL = Realized Profit Today + Current Floating Profit
            floating_profit = account.equity - account.balance
            active_daily_pnl = realized_profit_today + floating_profit
            
            start_of_day_balance = account.balance - realized_profit_today
            
            if start_of_day_balance <= 0: 
                return True
            
            # 1. Daily Drawdown Check (3% Rule)
            max_daily_dd = float(self.config.get('safety', {}).get('max_daily_drawdown_pct', 2.0))
            if active_daily_pnl < 0:
                daily_dd_pct = (abs(active_daily_pnl) / start_of_day_balance) * 100
                if daily_dd_pct >= max_daily_dd:
                    logger.critical(f"🛑 DRAWDOWN SHIELD: Daily loss is {daily_dd_pct:.2f}% (Limit {max_daily_dd}% limit reached). Blocking all Live Trades!")
                    return False
            
            # 2. Maximum Trailing Drawdown Check (6% Rule)
            # For a 10k account, hard floor is $9400. We block at $9450 (5.5%).
            # (Allows 0.5% margin of safety against slippage)
            if account.equity <= 9450.0:
                logger.critical(f"🛑 DRAWDOWN SHIELD: Equity (${account.equity:.2f}) is crossing 5.5% Max Trailing wall. Blocking all Live Trades!")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}")
            return True

    def place_mt5_trade(self, signal: Dict[str, Any]):
        """Place a live trade on MT5 terminal."""
        try:
            if not self.mt5:
                logger.error("MT5 not connected. Cannot place trade.")
                return False
                
            if not self.config.get('trading', {}).get('execute_trades', True):
                logger.info("Trading Disabled in config. Skipping MT5 execution.")
                return False

            # --- COMMODITY / BLOCKED SYMBOL / DIRECTIONAL SAFETY GATE ---
            from core.symbol_guard import is_symbol_blocked, is_direction_blocked
            if is_symbol_blocked(signal['symbol']):
                logger.critical(f"🛑 COMMODITY SHIELD: {signal['symbol']} is a blacklisted commodity. Blocking Live MT5 Trade execution!")
                signal['is_hidden'] = 1
                return False
            if is_direction_blocked(signal['symbol'], signal.get('signal')):
                logger.critical(f"🛑 DIRECTIONAL SHIELD: {signal['symbol']} {signal.get('signal')} is blacklisted by directional shield. Blocking Live MT5 Trade execution!")
                signal['is_hidden'] = 1
                return False

            # --- PROP FIRM DRAWDOWN & WEEKEND SAFETY GATE ---
            from core.guardrail import get_guardrail
            guard = get_guardrail()
            safety_status = guard.get_safety_status()
            if not safety_status['safe']:
                logger.error(f"BLOCK: {signal['symbol']} trade canceled by Safety Guardrail ({safety_status['reason']}). Saving as SHADOW instead.")
                signal['is_hidden'] = 1 # Force it into shadow log
                return False

            symbol = signal['symbol']
            action = signal['signal']

            # ── ABSOLUTE DUPLICATE LOCK: NEVER OPEN 2 POSITIONS ON THE SAME PAIR ──
            existing_positions = self.mt5.positions_get(symbol=symbol)
            if existing_positions:
                logger.warning(f"🛑 CRITICAL DEDUP LOCK: Position for {symbol} already exists in MT5 ({len(existing_positions)} open). REJECTING new trade.")
                return False

            price = signal['price_at_signal']
            sl = signal['sl_price']
            tp = signal['tp_price']
            
            # Calculate dynamic lots (0.5% risk)
            sl_pips = abs(price - sl) / self.inference_engine.data_engine.get_pip_value(symbol)
            lots = self._calculate_lot_size(symbol, sl_pips)
            
            # MT5 Order Type
            order_type = self.mt5.ORDER_TYPE_BUY if action == "BUY" else self.mt5.ORDER_TYPE_SELL
            
            # 🔍 Determine supported filling mode (Broker-Specific)
            filling_type = self.mt5.ORDER_FILLING_FOK
            symbol_info = self.mt5.symbol_info(symbol)
            if symbol_info:
                # Check bitmask for supported modes (1=FOK, 2=IOC)
                if (symbol_info.filling_mode & 1):
                    filling_type = self.mt5.ORDER_FILLING_FOK
                elif (symbol_info.filling_mode & 2):
                    filling_type = self.mt5.ORDER_FILLING_IOC
                else:
                    # Common retail fallback
                    filling_type = self.mt5.ORDER_FILLING_RETURN

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": 202404,  # APEX Magic Number
                "comment": f"APEX {signal.get('confidence_tier')}%",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }

            # Send order
            result = self.mt5.order_send(request)
            if not result or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                err_msg = result.comment if result else "Connection Timeout"
                err_code = result.retcode if result else "N/A"
                logger.error(f"MT5 ORDER FAILED: {err_msg} (Code: {err_code}) | Mode: {filling_type}")
                return None

                
            logger.info(f"LIVE TRADE PLACED: {symbol} {action} {lots} lots @ {price} | Ticket: {result.order}")
            return result.order
            
        except Exception as e:
            logger.error(f"Critical error in MT5 execution: {e}")
            return None

    def _is_duplicate_signal(self, symbol: str, last_candle_time: pd.Timestamp) -> bool:
        """
        Check if we already generated a signal for this specific candle.
        Prevents spamming signals when market is closed (data is static).
        """
        try:
            # Check DB for last signal
            recent = self.db.get_recent_signals(limit=1, symbol=symbol)
            if not recent:
                return False
                
            last_signal_ts_str = recent[0]['timestamp']
            last_signal_ts = pd.to_datetime(last_signal_ts_str)
            
            # Ensure timezone awareness compatibility
            if last_candle_time.tzinfo is None:
                last_candle_time = last_candle_time.tz_localize('UTC')
            if last_signal_ts.tzinfo is None:
                last_signal_ts = last_signal_ts.tz_localize('UTC')
                
            # If the last signal was generated AFTER the candle closed, we already saw this data.
            # Adding a small buffer (e.g., 1 minute) to allow for processing time differences
            if last_signal_ts > last_candle_time:
                return True
                
            return False
        except Exception as e:
            logger.warning(f"Deduplication check failed: {e}")
            return False

    def analyze_symbol(self, symbol: str, precalculated_result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze a single symbol and generate signal if criteria met.
        Only saves a new BUY/SELL signal if there's no existing active signal for this pair.
        Returns signal dict or None.
        """
        try:
            if precalculated_result is not None:
                result = precalculated_result
            else:
                # Use InferenceEngine for prediction
                # save_to_db=False because we handle DB saving here after deduplication
                result = self.inference_engine.predict_symbol(
                    symbol,
                    save_to_db=False,
                    win_rate=self.target_win_rate
                )
            
            if not result:
                return None
            
            if result.get('is_locked'):
                logger.info(f"🔒 {symbol}: Active trade lock propagated from InferenceEngine. Skipping scan.")
                return None

            # Note: model_trades gate removed — all loaded models are specialist models
            # Previously this silently blocked all signals when trades count wasn't populated
            
            signal = result['signal']
            new_tier = int(result.get('confidence_tier', 0))
            
            if signal in ('BUY', 'SELL'):
                from core.symbol_guard import is_direction_blocked
                if is_direction_blocked(symbol, signal):
                    logger.warning(f"🚫 DIRECTIONAL BLOCK: {symbol} {signal} is blacklisted by directional shield. Skipping live executive entry.")
                    result['is_hidden'] = 1
                    return None

                is_proven = bool(result.get('is_proven', False))
                
                # 1. Temporal Cooldown Check (Deduplication)
                # Check if we already generated a signal for this specific candle
                last_candle_time = result.get('timestamp_candle')
                if last_candle_time:
                    if self._is_duplicate_signal(symbol, pd.to_datetime(last_candle_time)):
                        logger.info(f"SKIP: {symbol}: Signal for this candle already exists in DB.")
                        return None

                # Query the database for the most recent actual BUY/SELL trade for this symbol
                # to prevent standard 'WAIT' signals from bypassing cooldown checks.
                recent_trade = None
                import sqlite3
                try:
                    with sqlite3.connect(self.db.db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        cur = conn.cursor()
                        cur.execute(
                            "SELECT * FROM signals WHERE symbol = ? AND signal IN ('BUY', 'SELL') ORDER BY timestamp DESC LIMIT 1",
                            (symbol,)
                        )
                        row = cur.fetchone()
                        if row:
                            recent_trade = dict(row)
                except Exception as db_err:
                    logger.error(f"Failed to query recent trades for cooldown: {db_err}")

                # Directional Cooldown: Check if the last trade was a WIN or LOSS
                last_time = None
                is_last_fail = False
                is_last_success = False
                is_last_active = False
                
                if recent_trade:
                    try:
                        # Helper to parse timezone-aware datetime consistently
                        ts_str = recent_trade['timestamp']
                        if ts_str.endswith('Z'):
                            ts_str = ts_str[:-1] + '+00:00'
                        last_time = datetime.fromisoformat(ts_str)
                        if last_time.tzinfo is None:
                            last_time = last_time.replace(tzinfo=timezone.utc)
                        outcome = recent_trade.get('outcome')
                        is_last_fail = (outcome == 'FAIL')
                        is_last_success = (outcome == 'SUCCESS')
                        # Ignore benched/shadow trades for active live lockups
                        is_last_active = (outcome in ('ACTIVE', 'NEW', 'PENDING') and not bool(recent_trade.get('is_hidden', 0)))
                    except Exception as parse_err:
                        logger.error(f"Error parsing cooldown timestamp for {symbol}: {parse_err}")

                if last_time:
                    # Active trade lock: prevent double-entry on same symbol
                    # (Cooldowns disabled — loss and win time-blocks removed)
                    if is_last_active:
                        logger.info(f"🛡️ ACTIVE TRADE LOCK: {symbol} has an active trade (ID {recent_trade.get('id')}) — blocking new trade.")
                        return None

                # REGIME CONFIDENCE GATE (DISABLED per user requirement: All 60%+ signals are authorized LIVE)
                # Previously downgraded ranging signals < 65% to shadow. Now all >= 60% signals go LIVE.
                pass

                # 2. Smart Stacking & Promotion Logic
                # Only count real trades (BUY/SELL) as active signals. Ignore 'WAIT'.
                active_signals = [s for s in self.db.get_active_signals(symbol=symbol, include_hidden=True) 
                                 if s['signal'] in ('BUY', 'SELL')]
                
                existing_live = any(not bool(s.get('is_hidden', 0)) for s in active_signals)
                new_tier = int(result.get('confidence_tier', 0))
                is_approved = (result.get('is_proven', False) and not bool(result.get('is_hidden', 0)))

                if active_signals:
                    # RULE 1: If a LIVE trade is open, all new signals for this direction MUST be SHADOW.
                    if existing_live and not bool(result.get('is_hidden', 0)):
                        logger.info(f"STACKING: {symbol} {signal}: Live trade active. Downgrading {new_tier}% signal to SHADOW for data tracking.")
                        result['is_hidden'] = 1
                    
                    # RULE 2: Deduplication — only deduplicate LIVE (is_hidden=0) trades at the same tier.
                    # Shadow 50% signals are record-only telemetry — they MUST NOT block each other.
                    # A new shadow signal in the same direction means conviction is persisting: expire
                    # the old one and save the fresh snapshot, so history stays accurate.
                    new_is_shadow = bool(result.get('is_hidden', 0))
                    new_confidence = float(result.get('confidence', 0))
                    new_price = float(result.get('price_at_signal', 0))
                    # ── BULLETPROOF LIVE POSITION SHIELD ──
                    # If there is already an active LIVE trade on this symbol in MT5, BLOCK any new entry!
                    if not new_is_shadow:
                        # 1. Check live positions directly in MT5 terminal
                        if self.mt5:
                            try:
                                existing_mt5_pos = self.mt5.positions_get(symbol=symbol)
                                if existing_mt5_pos:
                                    for p in existing_mt5_pos:
                                        if (signal == 'BUY' and p.type == 0) or (signal == 'SELL' and p.type == 1):
                                            logger.info(f"🛑 DEDUP MT5 SHIELD: {symbol} {signal} is already OPEN in MT5 (Ticket #{p.ticket}). Blocking duplicate order.")
                                            return None
                            except Exception as _mt5_err:
                                logger.warning(f"Error checking MT5 positions: {_mt5_err}")

                        # 2. Check active live signals in DB
                        for active in active_signals:
                            same_dir = active['signal'] == signal
                            active_is_live = not bool(active.get('is_hidden', 0))
                            if same_dir and active_is_live:
                                logger.info(f"🛑 DEDUP DB SHIELD: {symbol} {signal} has an active live signal (ID {active['id']}). Blocking duplicate entry.")
                                return None

                    for active in active_signals:
                        active_tier = active.get('confidence_tier', 0)
                        active_is_shadow = bool(active.get('is_hidden', 0))
                        try:
                            same_dir  = active['signal'] == signal
                            if same_dir:
                                if active_is_shadow and new_is_shadow:
                                    active_confidence = float(active.get('confidence', 0))
                                    conviction_delta = abs(new_confidence - active_confidence)

                                    if conviction_delta >= 0.01:
                                        logger.info(f"SHADOW ROLL: {symbol} {signal}: Conviction changed by {conviction_delta:.3f}. Expiring stale shadow (ID {active['id']}) and recording fresh snapshot.")
                                        self.db.update_signal_outcome(active['id'], 'EXPIRED', exit_reason='Shadow Rolled — Conviction Changed')
                                    else:
                                        logger.debug(f"SHADOW HOLD: {symbol} {signal}: Conviction unchanged ({conviction_delta:.4f}). Preserving original TP/SL for clean resolution.")
                                        return None
                                elif active_is_shadow and not new_is_shadow:
                                    logger.info(f"PROMOTION: {symbol} {signal} ({new_confidence*100:.1f}%): Upgrading shadow trade (ID {active['id']}) to LIVE entry.")
                                    self.db.update_signal_outcome(active['id'], 'EXPIRED', exit_reason='Promoted to Live')
                                else:
                                    logger.info(f"DEDUP: {symbol} {signal}: Live trade already active. Skipping duplicate.")
                                    return None
                        except (ValueError, TypeError):
                            continue

                    # RULE 3: Tier Promotion - If a benched shadow is running, and we hit a VALIDATED tier, allow it to go LIVE.
                    if not existing_live and is_approved:
                        from core.symbol_guard import is_direction_blocked
                        if not is_direction_blocked(symbol, signal):
                            logger.info(f"PROMOTION: {symbol} {signal} {new_tier}%: Approved tier found while shadows are active. Authorizing LIVE entry.")
                            result['is_hidden'] = 0
                        else:
                            logger.warning(f"🚫 PROMOTION BLOCKED: {symbol} {signal} is blacklisted by directional shield.")
                            result['is_hidden'] = 1

                # DEDUP GUARD: Final check to ensure we didn't just save an identical signal 
                # in the last few seconds (prevents triplicate race conditions)
                now = datetime.now(timezone.utc)
                last_sig_key = f"{symbol}_{signal}_{new_tier}"
                if last_sig_key in self._recent_signals:
                    last_time = self._recent_signals[last_sig_key]
                    if (now - last_time).total_seconds() < 10:
                        logger.info(f"DEDUP: {symbol} {signal} {new_tier}%: Just processed this signal < 10s ago. Discarding duplicate.")
                        return None

                # VALIDATION GUARD: Discard junk '0% Tier' signals
                if signal in ('BUY', 'SELL') and new_tier <= 0:
                    logger.warning(f"SKIP: {symbol} {signal}: Invalid Tier (0%). This happens during whitelist sync. Discarding.")
                    return None

            # ALWAYS persist the latest analysis outcome for the dashboard
            sig_id = self.db.save_signal(result)
            result['id'] = sig_id
            # Track in memory to block near-instant triplicates
            self._recent_signals[f"{symbol}_{signal}_{new_tier}"] = datetime.now(timezone.utc)

            # ── CRITICAL: Flush the final is_hidden flag back to DB ──────────────
            # inference.py may have saved is_hidden=0 (live), but the regime gate,
            # stacking rules, or promotion logic may have changed it in memory.
            # We must write the corrected value to DB NOW, before apex_connect.py
            # polls the DB and potentially executes a trade that should be shadow.
            if signal in ('BUY', 'SELL'):
                self.db.update_signal_hidden(sig_id, int(result.get('is_hidden', 0)))

            if signal in ('BUY', 'SELL'):
                from core.symbol_guard import is_symbol_blocked, is_direction_blocked
                if is_symbol_blocked(symbol) or is_direction_blocked(symbol, signal):
                    result['is_hidden'] = 1
                    self.db.update_signal_hidden(sig_id, 1)

                # 3. Certification Gate: Only alert and log as NEW if proven for MT5
                # and NOT hidden (Shadow Training)
                is_hidden = bool(result.get('is_hidden', False))
                
                if is_proven and not is_hidden:
                    log_label = "BUY" if signal == "BUY" else "SELL"
                    logger.info(
                        f"NEW CERTIFIED SIGNAL: {symbol} {log_label} @ {result['price_at_signal']:.5f} "
                        f"(Conf: {result['confidence']:.1%})"
                    )
                    
                    # LIVE TRADE EXECUTION — all regimes (regime-based model routing handled in inference.py)
                    regime = result.get('regime', 'NORMAL')
                    ticket = self.place_mt5_trade(result)
                    if ticket:
                        result['is_live'] = True
                        result['mt5_ticket'] = ticket
                        self.db.update_signal_ticket(result['id'], ticket)

                    # 2b. Broadcast to copy trading accounts (Multi-User)
                    try:
                        from scripts.multi_executor import execute_signal_for_all_users
                        execute_signal_for_all_users(result)
                        # Restore master terminal context after multi-account loop
                        mt5_conf = self.config.get('mt5', {})
                        if self.mt5 and mt5_conf.get('login'):
                            self.mt5.initialize(
                                login=int(mt5_conf['login']),
                                password=str(mt5_conf.get('password', '')),
                                server=str(mt5_conf.get('server', ''))
                            )
                    except Exception as _me:
                        logger.error(f"Multi-user copy execution error: {_me}")

                    # Send Telegram alert (Unless CRISIS)
                    if 'CRISIS' in str(regime).upper():
                        logger.warning(f"BLOCK: Telegram alert suppressed for {symbol} due to CRISIS regime.")
                    else:
                        self.notifier.send_signal_alert(result)
                elif is_hidden:
                    logger.info(f"SHADOW TRADE: {symbol} {signal} (Logged for certification history only)")
                    
                    # --- Notification Safety Gate ---
                    # NEVER send Telegram alerts if the market is in CRISIS
                    regime = result.get('regime', 'NORMAL')
                    if 'CRISIS' in str(regime).upper():
                        logger.warning(f"BLOCK: Telegram alert suppressed for {symbol} due to CRISIS regime.")
                        result['is_shadow_alert'] = False
                    elif self.notifier.notify_shadow:
                        result['is_shadow_alert'] = True
                        self.notifier.send_signal_alert(result)
                else:
                    logger.info(f"WATCH ONLY: {symbol} {signal} (Not yet certified in matrix)")
                
                # Update cooldown
                cooldown_key = f"{symbol}_{signal}"
                self._recent_signals[cooldown_key] = datetime.now(timezone.utc)
                return result
            else:
                # Update dashboard display with continuous convictions even when waiting
                result['outcome'] = 'N/A'
                self.db.save_signal(result)
            
            return None

            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            return None
    
    def run_scan(self, symbols: List[str]):
        """Execute a single market scan across all symbols."""
        # Unconditionally monitor outcomes first, before checking the safety gate
        self.monitor_active_signals()

        from core.guardrail import get_guardrail
        guard = get_guardrail()
        status = guard.get_safety_status()
        if not status['safe']:
            logger.warning(f"🛑 SAFETY HALT: {status['reason']} (Drawdown: {status['drawdown']:.1f}%) - Skipping Setup Scan.")
            return
            
        start_time = time.time()
        logger.info(f"--- MARKET SCAN STARTED: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} ---")
        logger.info(f"Scanning {len(symbols)} pairs for {self.target_win_rate} setups...")
        
        # Expire stale signals (>48h old) before scanning
        # DISABLED: User requested indefinite active signal retention until outcome
        # self.db.expire_stale_signals(max_age_hours=48)
        
        signals_generated = 0
        
        for i, symbol in enumerate(symbols):
            # Skip blocked symbols (poor win rate / temporarily disabled)
            if symbol in self._blocked_symbols:
                logger.debug(f"SKIP: {symbol} is in blocked_symbols list.")
                continue
            
            result = self.analyze_symbol(symbol)
            if result:
                signals_generated += 1
            
            # Stagger requests to avoid bursting the 8/min limit
            # 7.5s * 8 = 60s. This keeps us strictly within limits.
            if i < len(symbols) - 1:
                time.sleep(7.5)
        elapsed = time.time() - start_time
        logger.info(f"--- SCAN COMPLETE: {signals_generated} new signals in {elapsed:.1f}s ---")
        logger.info("")
    def check_and_execute_friday_exit(self, mt5_conn, broker_offset: int = 3) -> bool:
        """
        Friday Auto-Exit: Closes all open positions and resolves all active signals (live & shadow)
        30 minutes before market close on Friday (16:30 New York time, accounting for seasonal DST).
        """
        try:
            from core.market_hours import is_friday_auto_exit_time, get_ny_time
            
            if is_friday_auto_exit_time():
                ny_now = get_ny_time()
                logger.warning(f"🕒 Friday Exit Triggered: New York time is {ny_now.strftime('%Y-%m-%d %H:%M:%S %Z')} (30 min before market close). Closing all active signals.")
                
                # Fetch all ACTIVE signals from database
                active_signals = self.db.get_active_signals(include_hidden=True)
                if not active_signals:
                    logger.debug("Friday Exit: No active signals to resolve in DB.")
                    return False
                
                # Close live MT5 positions if connected
                if mt5_conn:
                    positions = mt5_conn.positions_get()
                    if positions:
                        logger.info(f"Friday Exit: Found {len(positions)} open MT5 positions to close.")
                        for p in positions:
                            logger.info(f"Friday Exit: Closing position {p.symbol} (Ticket: {p.ticket})")
                            tick_sym = mt5_conn.symbol_info_tick(p.symbol)
                            if not tick_sym:
                                logger.error(f"Cannot get tick for {p.symbol} — skipping MT5 close.")
                                continue
                                
                            close_price = tick_sym.bid if p.type == mt5_conn.POSITION_TYPE_BUY else tick_sym.ask
                            close_type = mt5_conn.ORDER_TYPE_SELL if p.type == mt5_conn.POSITION_TYPE_BUY else mt5_conn.ORDER_TYPE_BUY
                            
                            filling = mt5_conn.ORDER_FILLING_FOK
                            s_info = mt5_conn.symbol_info(p.symbol)
                            if s_info:
                                if (s_info.filling_mode & 1):
                                    filling = mt5_conn.ORDER_FILLING_FOK
                                elif (s_info.filling_mode & 2):
                                    filling = mt5_conn.ORDER_FILLING_IOC
                                else:
                                    filling = mt5_conn.ORDER_FILLING_RETURN
                                    
                            close_request = {
                                "action": mt5_conn.TRADE_ACTION_DEAL,
                                "symbol": p.symbol,
                                "volume": p.volume,
                                "type": close_type,
                                "position": p.ticket,
                                "price": close_price,
                                "deviation": 20,
                                "magic": 999000,
                                "comment": "Apex Friday Exit",
                                "type_time": mt5_conn.ORDER_TIME_GTC,
                                "type_filling": filling,
                            }
                            
                            res = mt5_conn.order_send(close_request)
                            if res and res.retcode == mt5_conn.TRADE_RETCODE_DONE:
                                logger.info(f"✅ Friday Exit: Closed {p.symbol} ticket {p.ticket} at {close_price}")
                            else:
                                comment = res.comment if res else "No response"
                                logger.error(f"❌ Friday Exit: Failed to close ticket {p.ticket}: {comment}")
                    else:
                        logger.info("Friday Exit: No open MT5 positions found.")

                # Resolve all active signals (both live and shadow) in DB
                for sig in active_signals:
                    symbol = sig['symbol']
                    sig_id = sig['id']
                    direction = sig['signal']
                    price_at_sig = sig.get('price_at_signal')

                    # Skip heartbeats / non-trade signals
                    if direction not in ('BUY', 'SELL') or not price_at_sig:
                        self.db.update_signal_outcome(sig_id, 'EXPIRED', exit_reason="Friday Auto-Exit (Non-trade)")
                        continue

                    # Fetch current price estimate
                    current_price = None
                    if mt5_conn:
                        tick = mt5_conn.symbol_info_tick(symbol)
                        if tick:
                            current_price = tick.bid if direction == 'BUY' else tick.ask

                    # Grade outcome:
                    # - If we got a live tick and price moved meaningfully → SUCCESS/FAIL
                    # - If no tick or price == entry (market closed, no data) → EXPIRED
                    min_pip = 0.0001
                    if current_price is not None and abs(current_price - price_at_sig) > min_pip:
                        profit = (current_price - price_at_sig) if direction == 'BUY' else (price_at_sig - current_price)
                        outcome = 'SUCCESS' if profit > 0 else 'FAIL'
                        reason = "Friday Auto-Exit (Terminated)"
                    else:
                        # Market closed or no real movement — don't penalise as FAIL
                        outcome = 'EXPIRED'
                        reason = "Friday Auto-Exit (Market Closed)"
                        current_price = current_price or price_at_sig

                    logger.info(f"🏁 Friday Exit Resolved: {symbol} ID {sig_id} -> {outcome} ({reason}) @ {current_price:.5f}")
                    self.db.update_signal_outcome(sig_id, outcome, exit_price=current_price, exit_reason=reason)
                return True
            return False
        except Exception as e:
            logger.error(f"Error in check_and_execute_friday_exit: {e}", exc_info=True)
            return False

    def monitor_active_signals(self):
        """
        High-Reliability Watchdog: Resolves active signals using Triple-Check logic:
        1. MT5 Ticket Status (If live trade exists)
        2. MT5 Live Ticks (Fastest resolution)
        3. Data Engine Fetch (Fallback)
        """
        active_signals = self.db.get_active_signals(include_hidden=True)
        
        # Initialize MT5 for the highest quality resolution
        from core.mt5_connector import get_mt5
        mt5_conn = get_mt5()

        # Determine broker timezone offset dynamically.
        # CRITICAL: Do NOT use symbol_info_tick().time for offset detection.
        # On Sunday market open, the last tick can still be Friday's stale price,
        # causing the watchdog to think it's Friday 23:59 and fire a false Friday Auto-Exit,
        # killing all freshly-generated Monday open signals.
        # Instead, use tick.time ONLY if it's fresh (< 2 hours old). Otherwise fall back to +3.
        broker_offset = 3.0  # Safe default: FTMO / EET summer (UTC+3)
        if mt5_conn:
            try:
                tick = mt5_conn.symbol_info_tick("EURUSD")
                if tick:
                    tick_dt = datetime.fromtimestamp(tick.time, timezone.utc)
                    staleness_hours = (datetime.now(timezone.utc) - tick_dt).total_seconds() / 3600
                    if staleness_hours < 2.0:
                        # Tick is fresh (market open) — safe to derive real offset
                        dynamic_offset = round((tick_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0)
                        broker_offset = dynamic_offset
                        logger.info(f"Watchdog: Live tick detected. MT5 broker offset = {broker_offset:+.1f}h (tick age: {staleness_hours:.1f}h)")
                    else:
                        logger.info(f"Watchdog: Tick is stale ({staleness_hours:.1f}h old — market closed). Using safe default offset of {broker_offset:+.1f}h.")
            except Exception as e:
                logger.warning(f"Watchdog: Failed to detect MT5 broker timezone offset: {e}")

        # Check and execute Friday exit before evaluating regular SL/TP outcomes
        if self.check_and_execute_friday_exit(mt5_conn, broker_offset):
            # If Friday exit triggered, it resolved all active signals. Refresh list.
            active_signals = self.db.get_active_signals(include_hidden=True)

        if not active_signals:
            return
            
        # ── MT5 DISCONNECTION GUARD ──────────────────────────────────────────
        # If MT5 is completely unreachable (auth failed, bridge down, account expired),
        # do NOT attempt to resolve any signals. Every check would fall through to 
        # the data-engine fallback which can produce false FAILs on stale data.
        # Keep all signals ACTIVE and wait for reconnection.
        if not mt5_conn:
            logger.warning("Watchdog: MT5 is unreachable. Skipping all signal resolution to prevent false FAILs. Signals remain ACTIVE.")
            return
            
        logger.info(f"Watchdog: Syncing {len(active_signals)} active signals with market reality...")
        resolutions_found = False
        
        for sig in active_signals:
            symbol = sig['symbol']
            sig_id = sig['id']
            ticket = sig.get('mt5_ticket')
            tp = sig.get('tp_price')
            sl = sig.get('sl_price')
            direction = sig['signal']
            
            # Safety Flush for corrupted data
            if not tp or not sl or tp == 0.0 or sl == 0.0:
                logger.warning(f"EXPIRED: {symbol} (ID {sig_id}) has missing levels. Flushing.")
                self.db.update_signal_outcome(sig_id, 'EXPIRED')
                resolutions_found = True
                continue

            outcome = None
            reason = ""
            current_price = 0.0

            # ── CHECK 1: MT5 Ticket Status (Highest Priority) ──────────
            if ticket and str(ticket).isdigit() and int(ticket) > 0:
                if mt5_conn:
                    try:
                        # Check if position is still open
                        pos = mt5_conn.positions_get(ticket=int(ticket))
                        if pos:
                            # Position is still open in MT5! Check if it has hit SL/TP on our side as a safety fallback.
                            tick = mt5_conn.symbol_info_tick(symbol)
                            if tick:
                                current_price = tick.bid if direction == 'BUY' else tick.ask
                                hit_tp = (direction == 'BUY' and current_price >= tp) or (direction == 'SELL' and current_price <= tp)
                                hit_sl = (direction == 'BUY' and current_price <= sl) or (direction == 'SELL' and current_price >= sl)
                                
                                if hit_tp or hit_sl:
                                    logger.warning(f"⚠️ SAFETY FALLBACK: Position {ticket} ({symbol}) hit {'TP' if hit_tp else 'SL'} but is still open in MT5. Manually closing.")
                                    close_type = mt5_conn.ORDER_TYPE_SELL if direction == 'BUY' else mt5_conn.ORDER_TYPE_BUY
                                    
                                    filling = mt5_conn.ORDER_FILLING_FOK
                                    s_info = mt5_conn.symbol_info(symbol)
                                    if s_info:
                                        if (s_info.filling_mode & 1):
                                            filling = mt5_conn.ORDER_FILLING_FOK
                                        elif (s_info.filling_mode & 2):
                                            filling = mt5_conn.ORDER_FILLING_IOC
                                        else:
                                            filling = mt5_conn.ORDER_FILLING_RETURN
                                            
                                    close_request = {
                                        "action": mt5_conn.TRADE_ACTION_DEAL,
                                        "symbol": symbol,
                                        "volume": pos[0].volume,
                                        "type": close_type,
                                        "position": int(ticket),
                                        "price": current_price,
                                        "deviation": 20,
                                        "magic": 202404,
                                        "comment": f"APEX SAFETY CLOSE ({'TP' if hit_tp else 'SL'})",
                                        "type_time": mt5_conn.ORDER_TIME_GTC,
                                        "type_filling": filling,
                                    }
                                    res = mt5_conn.order_send(close_request)
                                    if res and res.retcode == mt5_conn.TRADE_RETCODE_DONE:
                                        outcome = 'SUCCESS' if hit_tp else 'FAIL'
                                        reason = f"{'TP' if hit_tp else 'SL'} Hit (Safety Fallback Close)"
                                        logger.info(f"✅ Safety fallback close succeeded for ticket {ticket}")
                                    else:
                                        comment = res.comment if res else "No response"
                                        logger.error(f"❌ Safety fallback close failed for ticket {ticket}: {comment}")
                                        continue
                                else:
                                    logger.info(f"Position {ticket} ({symbol}) is still active in MT5. Keeping ACTIVE.")
                                    continue
                            else:
                                logger.info(f"Position {ticket} ({symbol}) is still active in MT5. Keeping ACTIVE (No Tick data).")
                                continue
                        else:
                            # Position closed in MT5! Find out why in history.
                            import MetaTrader5 as mt
                            hist = mt5_conn.history_deals_get(position=int(ticket))
                            if hist:
                                # Filter for exit deals (entry: 1=OUT, 2=INOUT, 3=OUT_BY)
                                exit_deals = [d for d in hist if getattr(d, 'entry', None) in (1, 2, 3)]
                                if exit_deals:
                                    last_exit = exit_deals[-1]
                                    current_price = last_exit.price
                                    # Sum the profits of all deals associated with this position
                                    total_profit = sum(getattr(d, 'profit', 0.0) for d in hist)
                                    outcome = 'SUCCESS' if total_profit > 0 else 'FAIL'
                                    reason = f"MT5 Native Close (Profit: ${total_profit:.2f})"
                                else:
                                    logger.info(f"MT5 Position {ticket} closed, but exit deals not available in history yet. Waiting.")
                                    continue
                            else:
                                logger.info(f"MT5 Position {ticket} closed, but history deals not available yet. Waiting.")
                                continue
                    except Exception as e:
                        logger.warning(f"MT5 Ticket check failed for {symbol}: {e}. Keeping ACTIVE to prevent premature resolution.")
                        continue
                else:
                    logger.warning(f"Live trade ticket {ticket} exists but MT5 is not connected. Keeping ACTIVE to prevent premature resolution.")
                    continue

            # ── CHECK 2: MT5 Live Ticks (Medium Priority) ──────────────
            if not outcome and mt5_conn:
                try:
                    tick = mt5_conn.symbol_info_tick(symbol)
                    if tick:
                        # Use BID for BUY exits, ASK for SELL exits
                        current_price = tick.bid if direction == 'BUY' else tick.ask
                        if direction == 'BUY':
                            if current_price >= tp: outcome, reason = 'SUCCESS', 'TP Hit (Live Tick)'
                            elif current_price <= sl: outcome, reason = 'FAIL', 'SL Hit (Live Tick)'
                        else: # SELL
                            if current_price <= tp: outcome, reason = 'SUCCESS', 'TP Hit (Live Tick)'
                            elif current_price >= sl: outcome, reason = 'FAIL', 'SL Hit (Live Tick)'
                except Exception as e:
                    logger.warning(f"MT5 Tick resolution failed for {symbol}: {e}")

            # ── CHECK 3: Data Engine Fetch (Fallback) ──────────────────
            if not outcome:
                try:
                    # Try 1m first for precision, fallback to 5m
                    for interval in ["1m", "5m"]:
                        df = self.inference_engine.data_engine.fetch(symbol, interval=interval, days=2, use_cache=False)
                        if df is not None and not df.empty:
                            sig_ts = pd.to_datetime(sig['timestamp'])
                            if sig_ts.tzinfo is None: sig_ts = sig_ts.tz_localize('UTC')
                            if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
                            else: df.index = df.index.tz_convert('UTC')
                            
                            # Shift MT5 time index to true UTC
                            if self.inference_engine.data_engine.provider.name == 'mt5' and broker_offset != 0:
                                df.index = df.index - pd.Timedelta(hours=broker_offset)
                            
                            relevant = df[df.index >= sig_ts]
                            if not relevant.empty:
                                if direction == 'BUY':
                                    tp_hits = relevant[relevant['high'] >= tp]
                                    sl_hits = relevant[relevant['low'] <= sl]
                                    tp_idx = tp_hits.index[0] if not tp_hits.empty else None
                                    sl_idx = sl_hits.index[0] if not sl_hits.empty else None
                                    
                                    if tp_idx and sl_idx:
                                        if tp_idx < sl_idx: outcome, reason = 'SUCCESS', f'TP Hit ({interval} data)'
                                        else: outcome, reason = 'FAIL', f'SL Hit ({interval} data)'
                                    elif tp_idx: outcome, reason = 'SUCCESS', f'TP Hit ({interval} data)'
                                    elif sl_idx: outcome, reason = 'FAIL', f'SL Hit ({interval} data)'
                                else: # SELL
                                    tp_hits = relevant[relevant['low'] <= tp]
                                    sl_hits = relevant[relevant['high'] >= sl]
                                    tp_idx = tp_hits.index[0] if not tp_hits.empty else None
                                    sl_idx = sl_hits.index[0] if not sl_hits.empty else None
                                    
                                    if tp_idx and sl_idx:
                                        if tp_idx < sl_idx: outcome, reason = 'SUCCESS', f'TP Hit ({interval} data)'
                                        else: outcome, reason = 'FAIL', f'SL Hit ({interval} data)'
                                    elif tp_idx: outcome, reason = 'SUCCESS', f'TP Hit ({interval} data)'
                                    elif sl_idx: outcome, reason = 'FAIL', f'SL Hit ({interval} data)'
                                
                                if outcome:
                                    current_price = relevant['close'].iloc[-1]
                                    break # Found outcome
                except Exception as e:
                    logger.error(f"Fallback resolution failed for {symbol}: {e}")

            # Final Actuation of Outcome
            if outcome:
                logger.info(f"🏁 RESOLVED: {symbol} ID {sig_id} -> {outcome} ({reason})")
                self.db.update_signal_outcome(sig_id, outcome, exit_price=current_price, exit_reason=reason)
                resolutions_found = True

        # If trades were resolved, trigger a micro-update of the performance matrix
        # This keeps the dashboard perfectly in sync with real-time shadow performance.
        if resolutions_found:
            logger.info("Resolutions detected. Triggering Performance Matrix micro-update...")
            try:
                from core.performance_gate import PerformanceGate
                gate = PerformanceGate()
                gate.recompute_from_db(lookback_days=14)
                gate.save_whitelist()
                logger.info("Performance Matrix synchronization complete.")
            except Exception as e:
                logger.error(f"Real-time matrix update failed: {e}")
    
    def run_continuous(self, symbols: Optional[List[str]] = None):
        """
        Run continuous market monitoring.
        
        Args:
            symbols: List of symbols to monitor (None = all configured)
        """
        if symbols is None:
            symbols = self.inference_engine.data_engine.get_all_pairs()
        
        logger.info(f"Starting continuous monitoring: {len(symbols)} pairs")
        logger.info(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        logger.info(f"Target Win Rate: {self.target_win_rate}")
        logger.info("")
        
        try:
            while True:
                self._run_daily_maintenance()
                
                # ── HEARTBEAT (Start of cycle) ──────────────
                # Written BEFORE the scan so dashboard shows ACTIVE immediately.
                # The scan can take 5-15 min — we don't want to look stalled.
                self.db.save_heartbeat()
                
                self.run_scan(symbols)
                
                # Sleep until next scan
                sleep_time = self.scan_interval_minutes * 60
                next_scan = datetime.now(timezone.utc) + timedelta(seconds=sleep_time)
                logger.info(f"Next scan: {next_scan.strftime('%H:%M:%S')} ({self.scan_interval_minutes} min)")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("="*70)
            logger.info("EXECUTIVE ENGINE STOPPED BY USER")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Executive engine crashed: {e}", exc_info=True)
            raise


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Executive Engine - Production Background Worker")
    parser.add_argument('--win-rate', type=str, default='90%', help='Target Win Rate e.g. 90%')
    parser.add_argument('--interval', type=int, default=15, help='Scan interval in minutes (default: 15)')
    parser.add_argument('--symbols', nargs='+', default=None, help='Specific symbols to monitor')
    
    args = parser.parse_args()
    
    # Initialize and run
    engine = ExecutiveEngine(
        target_win_rate=args.win_rate,
        scan_interval_minutes=args.interval
    )
    
    engine.run_continuous(symbols=args.symbols)
