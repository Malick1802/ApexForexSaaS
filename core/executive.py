# =============================================================================
# Executive - Production Background Worker
# =============================================================================
"""
Autonomous signal generation engine that:
- Polls TwelveData API every 15 minutes
- Respects free tier limit (8 requests/minute)
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


# Rate limiting is now handled centrally in data_pipeline/providers/twelvedata_provider.py


# =============================================================================
# =============================================================================
# Executive Engine
# =============================================================================

class ExecutiveEngine:
    """
    Production background worker for autonomous signal generation.
    
    Features:
    - 15-minute scanning interval
    - TwelveData rate limiting (via InferenceEngine's DataEngine)
    - High-Confidence "Apex" signals (Default)
    - Telegram alerts
    - Full activity logging
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        target_win_rate: str = "70%",  # Set to user default (institutional floor)
        scan_interval_minutes: int = 5
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
        from core.mt5_connector import get_mt5
        self.mt5 = get_mt5()
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
        logger.info("="*70)

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

            # --- PROP FIRM DRAWDOWN SAFETY GATE ---
            if not self._check_drawdown_limits():
                logger.error(f"BLOCK: {signal['symbol']} trade canceled by Drawdown Safety Shield. Saving as SHADOW instead.")
                signal['is_hidden'] = 1 # Force it into shadow log
                return False

            symbol = signal['symbol']
            action = signal['signal']
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
                return False

                
            logger.info(f"LIVE TRADE PLACED: {symbol} {action} {lots} lots @ {price}")
            return True
            
        except Exception as e:
            logger.error(f"Critical error in MT5 execution: {e}")
            return False

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

    def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single symbol and generate signal if criteria met.
        Only saves a new BUY/SELL signal if there's no existing active signal for this pair.
        Returns signal dict or None.
        """
        try:
            # Use InferenceEngine for prediction
            # save_to_db=False because we handle DB saving here after deduplication
            result = self.inference_engine.predict_symbol(
                symbol,
                save_to_db=False,
                win_rate=self.target_win_rate
            )
            
            if not result:
                return None
            
            # Note: model_trades gate removed — all loaded models are specialist models
            # Previously this silently blocked all signals when trades count wasn't populated
            
            signal = result['signal']
            
            if signal in ('BUY', 'SELL'):
                is_proven = bool(result.get('is_proven', False))
                
                # 1. Temporal Cooldown Check (Deduplication)
                # Check if we already generated a signal for this specific candle
                last_candle_time = result.get('timestamp_candle')
                if last_candle_time:
                    if self._is_duplicate_signal(symbol, pd.to_datetime(last_candle_time)):
                        logger.info(f"SKIP: {symbol}: Signal for this candle already exists in DB.")
                        return None

                # Directional Cooldown: Only block if the NEW signal matches the LAST one
                cooldown_key = f"{symbol}_{signal}"
                last_time = self._recent_signals.get(cooldown_key)
                
                # If not in memory, check DB (persistent cooldown across restarts)
                if not last_time:
                    recent = self.db.get_recent_signals(limit=1, symbol=symbol)
                    if recent and recent[0]['signal'] == signal:
                        try:
                            last_time = pd.to_datetime(recent[0]['timestamp'])
                            if last_time.tzinfo is None:
                                last_time = last_time.tz_localize('UTC')
                            # Populate memory cache
                            self._recent_signals[cooldown_key] = last_time
                        except Exception:
                            pass

                if last_time:
                    elapsed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
                    if elapsed < self._cooldown_minutes:
                        logger.info(f"SKIP: {symbol} {signal}: Cooldown active ({elapsed:.1f}/{self._cooldown_minutes} min).")
                        return None

                # 1b. Loss Cooldown Check (Anti-Revenge Trading)
                # If the last resolved signal for this symbol was a FAIL, block new entries for 4 hours.
                if symbol not in self._loss_cooldowns:
                    # Lazily populate from DB on first encounter
                    recent_fail = self.db.get_recent_signals(limit=1, symbol=symbol)
                    if recent_fail and recent_fail[0].get('outcome') == 'FAIL':
                        try:
                            fail_ts = datetime.fromisoformat(recent_fail[0]['timestamp'])
                            if fail_ts.tzinfo is None:
                                fail_ts = fail_ts.replace(tzinfo=timezone.utc)
                            self._loss_cooldowns[symbol] = fail_ts
                        except Exception:
                            pass

                if symbol in self._loss_cooldowns:
                    elapsed_loss = (datetime.now(timezone.utc) - self._loss_cooldowns[symbol]).total_seconds() / 60
                    if elapsed_loss < self._loss_cooldown_minutes:
                        logger.info(f"🛡️ LOSS COOLDOWN: {symbol}: SL hit {elapsed_loss:.0f} min ago — blocked for {self._loss_cooldown_minutes - elapsed_loss:.0f} more min (4h anti-revenge guard).")
                        return None
                    else:
                        # Cooldown expired — remove it
                        del self._loss_cooldowns[symbol]

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
                    
                    # RULE 2: Deduplication - Don't save the EXACT same tier if it's already active.
                    for active in active_signals:
                        active_tier = active.get('confidence_tier', 0)
                        # Use int(float()) to safely convert "90" or "90.0" strings/numbers to 90
                        try:
                            if active['signal'] == signal and int(float(active_tier)) == int(new_tier):
                                logger.info(f"DEDUP: {symbol} {signal} {new_tier}%: Already active. Skipping duplicate.")
                                return None
                        except (ValueError, TypeError):
                            continue

                    # RULE 3: Tier Promotion - If a benched shadow is running, and we hit a VALIDATED tier, allow it to go LIVE.
                    if not existing_live and is_approved:
                        logger.info(f"PROMOTION: {symbol} {signal} {new_tier}%: Approved tier found while shadows are active. Authorizing LIVE entry.")
                        result['is_hidden'] = 0

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
            self.db.save_signal(result)
            # Track in memory to block near-instant triplicates
            self._recent_signals[f"{symbol}_{signal}_{new_tier}"] = datetime.now(timezone.utc)

            if signal in ('BUY', 'SELL'):
                # 3. Certification Gate: Only alert and log as NEW if proven for MT5
                # and NOT hidden (Shadow Training)
                is_hidden = bool(result.get('is_hidden', False))
                
                if is_proven and not is_hidden:
                    log_label = "BUY" if signal == "BUY" else "SELL"
                    logger.info(
                        f"NEW CERTIFIED SIGNAL: {symbol} {log_label} @ {result['price_at_signal']:.5f} "
                        f"(Conf: {result['confidence']:.1%})"
                    )
                    
                    # LIVE TRADE EXECUTION
                    if self.place_mt5_trade(result):
                        result['is_live'] = True
                    
                    # Send Telegram alert (Unless CRISIS)
                    regime = result.get('regime', 'NORMAL')
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
        
        # Monitor outcomes after each scan
        self.monitor_active_signals()
        logger.info("")
    
    def monitor_active_signals(self):
        """
        High-Reliability Watchdog: Resolves active signals using Triple-Check logic:
        1. MT5 Ticket Status (If live trade exists)
        2. MT5 Live Ticks (Fastest resolution)
        3. Data Engine Fetch (Fallback)
        """
        active_signals = self.db.get_active_signals(include_hidden=True)
        if not active_signals:
            return
            
        logger.info(f"Watchdog: Syncing {len(active_signals)} active signals with market reality...")
        resolutions_found = False
        
        # Initialize MT5 for the highest quality resolution
        from core.mt5_connector import get_mt5
        mt5_conn = get_mt5()
        
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
            if mt5_conn and ticket:
                try:
                    # Check if position is still open
                    pos = mt5_conn.positions_get(ticket=int(ticket))
                    if not pos:
                        # Position closed in MT5! Find out why in history.
                        import MetaTrader5 as mt
                        from datetime import datetime, timedelta
                        hist = mt5_conn.history_deals_get(ticket=int(ticket))
                        if hist:
                            deal = hist[-1]
                            current_price = deal.price
                            profit = deal.profit
                            outcome = 'SUCCESS' if profit > 0 else 'FAIL'
                            reason = f"MT5 Native Close (Profit: ${profit:.2f})"
                        else:
                            # Fallback if history deal not found yet
                            outcome = 'SUCCESS' # Optimistic until proven otherwise
                            reason = "MT5 Position Closed (History pending)"
                except Exception as e:
                    logger.warning(f"MT5 Ticket check failed for {symbol}: {e}")

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
                            
                            relevant = df[df.index >= sig_ts]
                            if not relevant.empty:
                                if direction == 'BUY':
                                    if (relevant['high'] >= tp).any(): outcome, reason = 'SUCCESS', f'TP Hit ({interval} data)'
                                    elif (relevant['low'] <= sl).any(): outcome, reason = 'FAIL', f'SL Hit ({interval} data)'
                                else: # SELL
                                    if (relevant['low'] <= tp).any(): outcome, reason = 'SUCCESS', f'TP Hit ({interval} data)'
                                    elif (relevant['high'] >= sl).any(): outcome, reason = 'FAIL', f'SL Hit ({interval} data)'
                                
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
