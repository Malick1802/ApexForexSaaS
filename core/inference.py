# =============================================================================
# Inference Engine - Automatic Signal Generation
# =============================================================================
"""
Background inference engine that monitors currency pairs and generates
trading signals automatically with Entry/Stop-Loss/Take-Profit levels.

Features:
- Multi-pair monitoring with configurable intervals
- Automatic TP/SL calculation based on pip values
- Signal deduplication to avoid spam
- Database persistence
- Binary model support (BUY/SELL classifiers)
"""

import os
import sys
import logging
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import OrderedDict
import gc
import joblib

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from .database import SignalDatabase
from .performance_gate import get_performance_gate
# from core.bayesian_engine import get_bayesian_engine
from .regime_detector import get_detector
from .calibration import get_calibration_manager
from .notifications import NotificationManager
from .mt5_connector import get_mt5

try:
    from tensorflow import keras
    _TF_AVAILABLE = True
    _TF_ERROR = None
except ImportError as e:
    keras = None
    _TF_AVAILABLE = False
    _TF_ERROR = str(e)


logger = logging.getLogger(__name__)

# Standard Project Root (Robust Anchor)
def get_project_root():
    current_file = Path(__file__).resolve()
    # Check if we are in core/ or root
    if current_file.parent.name == "core":
        return current_file.parent.parent
    return current_file.parent

PROJECT_ROOT = get_project_root()


# Golden Signal Quality Gate — minimum certified backtest win rate
# Models below this threshold are blocked from generating live signals.
GOLDEN_MIN_WIN_RATE = 0.60  # 60% minimum

class InferenceEngine:
    """
    Autonomous inference engine for generating trading signals.
    
    Usage:
        engine = InferenceEngine()
        signal = engine.predict_symbol("EURUSD")
        engine.run_continuous(interval_minutes=5)
    """
    
    def __init__(
        self,
        model_dir: str = "models/binary",
        config_path: str = "config.yaml",
        confidence_threshold: float = 0.92,
        cooldown_minutes: int = 15  # Reduced to 15-min for dynamic trading (User requested)
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing trained models
            config_path: Path to config.yaml
            confidence_threshold: Minimum confidence for signal generation
            cooldown_minutes: Anti-spam cooldown in minutes (default: 15)
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self._signal_cooldown_minutes = cooldown_minutes
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        self.notifier = NotificationManager()
        self.global_engineer = GlobalFeatureEngineer()
        self.db = SignalDatabase()
        self._regime_detector = get_detector()
        
        # Cache for loaded models (LRU Implementation)
        self._model_cache = OrderedDict()
        self._max_cached_models = 10 
        self._global_data_cache = None
        self._last_global_update = None
        
        self._recent_signals: Dict[str, datetime] = {}
        self._signal_cooldown_minutes = 240  # 4-hour cooldown: don't repeat same pair within 4 hours
        
        # ── Calibration & Performance Gates ───────────────────────────────────────
        self.calibrator = get_calibration_manager()
        self.perf_gate = get_performance_gate()

        if not _TF_AVAILABLE:
            raise ImportError(
                f"TensorFlow initialization failed. The application cannot load AI models.\n"
                f"Likely missing Microsoft C++ Redistributable on this VM.\n"
                f"Original Error: {_TF_ERROR}\n"
                f"Fix: Download and install from https://aka.ms/vs/17/release/vc_redist.x64.exe"
            )

        logger.info(f"InferenceEngine initialized with model_dir={model_dir}")

        
    def calculate_lots_precision(self, symbol: str, entry: float, sl: float) -> float:
        """
        Prop-Grade Lot Calculation for GetLeveraged.com (1:30 Leverage).
        Ensures 0.5% risk on BALANCE (to protect trailing drawdown)
        and caps lots based on 1:30 physical margin limit.
        """
        try:
            mt5_config = self.config.get('mt5', {})
            risk_pct = mt5_config.get('risk_value', 0.5)
            max_leverage = mt5_config.get('max_trade_leverage', 30)
            
            _mt5 = get_mt5()
            if not _mt5:
                return 0.01

            account = _mt5.account_info()
            if not account:
                return 0.01

            # Use BALANCE instead of EQUITY to avoid trailing drawdown spikes
            risk_amount = account.balance * (risk_pct / 100.0)
            
            symbol_info = _mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01

            # 1. Calculate Risk-Based Lots
            tick_size = symbol_info.trade_tick_size
            price_dist = abs(entry - sl)
            dist_in_ticks = price_dist / tick_size
            tick_value = symbol_info.trade_tick_value
            
            if dist_in_ticks <= 0 or tick_value <= 0:
                return 0.01
                
            loss_per_lot = dist_in_ticks * tick_value
            risk_lots = risk_amount / loss_per_lot

            # 2. Calculate Margin-Limited Maximum (Leverage Cap Safe)
            # Use MT5 Native Margin Calc for exact currency/leverage handling
            margin_per_lot = _mt5.order_calc_margin(_mt5.ORDER_TYPE_BUY, symbol, 1.0, entry)
            
            if not margin_per_lot:
                # Fallback to conservative estimate if calc fails
                notional_per_lot = entry * symbol_info.trade_contract_size
                margin_per_lot = notional_per_lot / max_leverage
                
            # Max Buying Power is our whole balance (but we use 90% for safety)
            max_margin_lots = (account.balance * 0.9) / margin_per_lot

            # 3. Take the Smallest (Risk vs. Reality)
            final_lots = min(risk_lots, max_margin_lots)
            
            # Normalize to Volume Step
            step = symbol_info.volume_step
            final_lots = round(final_lots / step) * step
            
            return max(symbol_info.volume_min, min(symbol_info.volume_max, final_lots))
        except Exception as e:
            logger.error(f"Margin-safe lot calculation failed: {e}")
            return 0.01

    def _get_correlated_assets(self, symbol: str) -> List[dict]:
        """Get correlated assets config for a symbol."""
        if 'currency_pairs' not in self.config:
            return []
            
        categories = ['majors', 'minors', 'crosses']
        for category in categories:
            pairs = self.config['currency_pairs'].get(category, [])
            for pair in pairs:
                if pair['symbol'] == symbol:
                    return pair.get('correlated_assets', [])
        return []
    
    def load_binary_models(self, symbol: str) -> Optional[Dict]:
        """
        Load BUY and SELL binary classifiers for a symbol.
        Checks multiple directories: specialist (priority), binary.
        
        Returns:
            Dict with 'buy_model', 'sell_model', 'scaler' or None if not found
        """
        if symbol in self._model_cache:
            # Move to end to mark as recently used
            models = self._model_cache.pop(symbol)
            self._model_cache[symbol] = models
            return models
        
        # Try specialist directory ONLY for certified models
        base_dirs = [
            str(PROJECT_ROOT / "models" / "specialist"),
            "models/specialist"
        ]
        
        for base_dir in base_dirs:
            try:
                buy_path = Path(base_dir) / symbol / "BUY" / "model.keras"
                sell_path = Path(base_dir) / symbol / "SELL" / "model.keras"
                scaler_path = Path(base_dir) / symbol / "BUY" / "scaler.joblib"
                
                if not (buy_path.exists() and sell_path.exists() and scaler_path.exists()):
                    continue
                
                # Load BOTH BUY and SELL configs for per-direction metadata
                buy_trades = 0
                sell_trades = 0
                buy_threshold = 0.5
                sell_threshold = 0.5
                buy_win_rate = 0.0
                sell_win_rate = 0.0
                
                buy_config_path = Path(base_dir) / symbol / "BUY" / "config.json"
                if buy_config_path.exists():
                    try:
                        with open(buy_config_path, 'r') as f:
                            bc = json.load(f)
                            buy_trades = bc.get('trades', 0)
                            buy_threshold = bc.get('threshold', 0.5)
                            buy_win_rate = bc.get('win_rate', 0.0)
                    except: pass
                
                sell_config_path = Path(base_dir) / symbol / "SELL" / "config.json"
                if sell_config_path.exists():
                    try:
                        with open(sell_config_path, 'r') as f:
                            sc = json.load(f)
                            sell_trades = sc.get('trades', 0)
                            sell_threshold = sc.get('threshold', 0.5)
                            sell_win_rate = sc.get('win_rate', 0.0)
                    except: pass
                
                # ── Golden Quality Gate ──────────────────────────────
                # Block models that don't meet the minimum win rate.
                # A model must have BOTH directions above the floor,
                # or the failing direction is masked out.
                buy_qualified = buy_win_rate >= GOLDEN_MIN_WIN_RATE and buy_trades >= 10
                sell_qualified = sell_win_rate >= GOLDEN_MIN_WIN_RATE and sell_trades >= 10
                
                if not buy_qualified and not sell_qualified:
                    logger.warning(
                        f"QUALITY GATE BLOCKED {symbol}: "
                        f"BUY WR={buy_win_rate:.1%} ({buy_trades}t), "
                        f"SELL WR={sell_win_rate:.1%} ({sell_trades}t) "
                        f"— both below {GOLDEN_MIN_WIN_RATE:.0%} floor"
                    )
                    continue  # Skip this model entirely
                
                if not buy_qualified:
                    logger.info(f"QUALITY GATE: {symbol} BUY masked (WR={buy_win_rate:.1%}, {buy_trades}t)")
                if not sell_qualified:
                    logger.info(f"QUALITY GATE: {symbol} SELL masked (WR={sell_win_rate:.1%}, {sell_trades}t)")
                
                models = {
                   'buy_model': keras.models.load_model(str(buy_path)) if buy_qualified else None,
                    'sell_model': keras.models.load_model(str(sell_path)) if sell_qualified else None,
                    'scaler': joblib.load(str(scaler_path)),
                    'model_type': 'binary',
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'model_trades': max(buy_trades, sell_trades),
                    'buy_threshold': max(buy_threshold, 0.70),
                    'sell_threshold': max(sell_threshold, 0.70),
                    'buy_win_rate': buy_win_rate,
                    'sell_win_rate': sell_win_rate,
                    'buy_qualified': buy_qualified,
                    'sell_qualified': sell_qualified,
                }
                
                # LRU Check & Memory Cleanup
                if len(self._model_cache) >= self._max_cached_models:
                    oldest_key, _ = self._model_cache.popitem(last=False)
                    logger.info(f"🧹 LRU Eviction: {oldest_key}. Reclaiming memory...")
                    self._cleanup_memory()

                self._model_cache[symbol] = models
                logger.info(
                    f"Loaded GOLDEN models for {symbol} "
                    f"(BUY: {'ACTIVE' if buy_qualified else 'BLOCKED'} {buy_win_rate:.1%}, "
                    f"SELL: {'ACTIVE' if sell_qualified else 'BLOCKED'} {sell_win_rate:.1%})"
                )
                return models
                
            except Exception as e:
                logger.debug(f"Could not load from {base_dir}/{symbol}: {e}")
                continue
        
        logger.warning(f"Binary models not found for {symbol}")
        return None

    def load_expert_model(self, symbol: str, win_rate: int) -> Optional[Dict]:
        """
        Load Expert Model for a specific win rate (60, 70, 80, 90, 95).
        Location: models/{symbol}/{win_rate}/[BUY|SELL]
        """
        cache_key = f"{symbol}_{win_rate}"
        if cache_key in self._model_cache:
            # Move to end (LRU)
            models = self._model_cache.pop(cache_key)
            self._model_cache[cache_key] = models
            return models
            
        try:
            # ROBUST PATH SEARCH
            # 1. Primary: PROJECT_ROOT based
            logger.info(f"DEBUG PATH: PROJECT_ROOT={PROJECT_ROOT}")
            paths_to_check = [PROJECT_ROOT / "models" / symbol / str(win_rate)]
            
            # 2. Secondary: Relative to CWD
            paths_to_check.append(Path("models") / symbol / str(win_rate))
            
            # 3. Tertiary: Absolute brute force (if running from different root)
            # This handles cases where PROJECT_ROOT might be mis-detected
            if "ApexForexSaaS" in str(PROJECT_ROOT):
                pass 
            else:
                 # Try to find ApexForexSaaS in CWD parents
                 cwd = Path(os.getcwd())
                 if "ApexForexSaaS" in cwd.name:
                     paths_to_check.append(cwd / "models" / symbol / str(win_rate))

            base_project_dir = None
            for p in paths_to_check:
                if (p / "BUY" / "model.keras").exists():
                    base_project_dir = p
                    break
            
            if base_project_dir is None:
                logger.warning(f"Expert models not found for {symbol} @ {win_rate}%. Checked: {[str(p) for p in paths_to_check]}")
                return None

            buy_path = base_project_dir / "BUY" / "model.keras"
            sell_path = base_project_dir / "SELL" / "model.keras"
            # scaler is in BUY folder for this model iteration
            scaler_path = base_project_dir / "BUY" / "scaler.joblib"
            
            # Configs for thresholds
            buy_config_path = base_project_dir / "BUY" / "config.json"
            sell_config_path = base_project_dir / "SELL" / "config.json"
                
            # Load Configs to get thresholds and volume
            buy_threshold = 0.5
            sell_threshold = 0.5
            trades_count = 0
            
            if buy_config_path.exists():
                with open(buy_config_path, 'r') as f:
                    config_data = json.load(f)
                    buy_threshold = config_data.get('threshold', 0.5)
                    trades_count = max(trades_count, config_data.get('trades', 0))
            
            if sell_config_path.exists():
                with open(sell_config_path, 'r') as f:
                    config_data = json.load(f)
                    sell_threshold = config_data.get('threshold', 0.5)
                    trades_count = max(trades_count, config_data.get('trades', 0))

            models = {
                'buy_model': keras.models.load_model(str(buy_path)),
                'sell_model': keras.models.load_model(str(sell_path)),
                'scaler': joblib.load(str(scaler_path)),
                'model_type': 'expert', # Treated like binary but with custom thresholds
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'model_trades': trades_count
            }
            
            # LRU Check & Memory Cleanup
            if len(self._model_cache) >= self._max_cached_models:
                oldest_key, _ = self._model_cache.popitem(last=False)
                logger.info(f"🧹 LRU Eviction: {oldest_key}. Reclaiming memory...")
                self._cleanup_memory()

            self._model_cache[cache_key] = models
            logger.info(f"Loaded EXPERT models for {symbol} (Target: {win_rate}%, Vol: {trades_count})")
            return models
            
        except Exception as e:
            logger.error(f"Failed to load expert model {symbol}/{win_rate}: {e}")
            return None
    
    def _cleanup_memory(self):
        """Proactively clear Keras session and trigger garbage collection."""
        try:
            keras.backend.clear_session()
            gc.collect()
            logger.info("♻️ Explicit memory reclamation completed.")
        except Exception as e:
            logger.warning(f"Memory cleanup error: {e}")

    def load_enhanced_model(self, symbol: str) -> Optional[Dict]:
        """
        Load enhanced 3-class model (BUY/SELL/WAIT).
        Checks multiple directories: enhanced, specialist, trained.
        
        Returns:
            Dict with 'model', 'scaler' or None if not found
        """
        # Try multiple model directories
        model_dirs = ["models/enhanced", "models/specialist", "models/trained"]
        
        for base_dir in model_dirs:
            model_path = Path(base_dir) / symbol / "model.keras"
            scaler_path = Path(base_dir) / symbol / "scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    # Try to load trades volume
                    trades_count = 0
                    config_path = Path(base_dir) / symbol / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                trades_count = json.load(f).get('trades', 0)
                        except: pass

                    models = {
                        'model': keras.models.load_model(str(model_path)),
                        'scaler': joblib.load(str(scaler_path)),
                        'model_type': 'enhanced',
                        'model_trades': trades_count
                    }
                    
                    logger.info(f"Loaded 3-class model for {symbol} (Vol: {trades_count})")
                    return models
                    
                except Exception as e:
                    logger.error(f"Failed to load model for {symbol} from {base_dir}: {e}")
                    continue
        
        return None

        return None

    def load_models(self, symbol: str, win_rate: Optional[int] = None) -> Optional[Dict]:
        """
        Unified model loader - STRICT SPECIALIST MODE.
        Only loads models from the certified specialist fleet (models/specialist/).
        Standard pool and non-certified expert models are disabled.
        """
        return self.load_binary_models(symbol)
    
    def calculate_tp_sl(
        self,
        symbol: str,
        signal: str,
        entry_price: float,
        atr_pips: Optional[float] = None,
        spread_pips: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate Take Profit and Stop Loss levels.
        Supports both static (default) and dynamic (ATR-based) levels.
        
        Args:
            symbol: Currency pair
            signal: "BUY" or "SELL"
            entry_price: Entry price level
            atr_pips: Optional ATR in pips for dynamic calculation
            
        Returns:
            Dict with tp_price, sl_price, tp_pips, sl_pips
        """
        # Get trading config
        trading_config = self.config.get('trading', {})
        
        # Default/Static values from config
        min_sl_pips = trading_config.get('stop_loss_pips', 25)
        rr_ratio = trading_config.get('risk_reward_ratio', 1.5)
        
        # Determine pip type
        is_gold = 'XAU' in symbol or 'GOLD' in symbol
        if is_gold:
            pip_type = 'gold'
        else:
            pip_type = 'jpy' if 'JPY' in symbol else 'standard'
            
        pip_values = trading_config.get('pip_values', {})
        pip_value = pip_values.get(pip_type, 0.01 if pip_type == 'gold' else 0.0001)
        
        spread_dist = spread_pips * pip_value
        
        # Rule 1: SL = max(ATR * 1.5, min_sl_pips) + spread
        if is_gold:
            # Special floor for Gold to prevent over-leveraging on tight stops
            min_sl_floor = trading_config.get('gold_min_sl_pips', 100)
        else:
            min_sl_floor = min_sl_pips

        if atr_pips is not None and atr_pips > 0:
            dynamic_sl = atr_pips * 1.5
            base_sl = max(min_sl_floor, dynamic_sl)
        else:
            base_sl = min_sl_floor
            
        sl_pips = base_sl + spread_pips
        
        # Rule 2: TP = (base_sl * RR) + spread
        tp_pips = (base_sl * rr_ratio) + spread_pips
        
        if signal == "BUY":
            tp_price = entry_price + (tp_pips * pip_value)
            sl_price = entry_price - (sl_pips * pip_value)
        else:  # SELL
            tp_price = entry_price - (tp_pips * pip_value)
            sl_price = entry_price + (sl_pips * pip_value)
        
        return {
            'tp_price': round(tp_price, 5),
            'sl_price': round(sl_price, 5),
            'tp_pips': int(tp_pips),
            'sl_pips': int(sl_pips),
            'pip_value': pip_value
        }
    

    
    def _is_data_stale(self, last_candle_time: pd.Timestamp) -> bool:
        """
        Check if data is too old (e.g., market closed).
        
        Logic:
        1. If > 2 hours old, it's stale (allowing for 1h candle + 1h delay).
        2. If it's the weekend (Sat/Sun), stricter checks apply.
        """
        try:
            # 1. Standard Comparison (UTC vs UTC)
            now_utc = pd.Timestamp.now(tz='UTC')
            
            if last_candle_time.tzinfo is None:
                # Assume UTC if naive (Standard for yfinance/twelvedata in this pipeline)
                last_candle_aware = last_candle_time.tz_localize('UTC')
            else:
                last_candle_aware = last_candle_time.tz_convert('UTC')
                
            diff = now_utc - last_candle_aware
            hours_diff = diff.total_seconds() / 3600.0
            
            # Debug log (Temporary enabled for diagnosis)
            # logger.info(f"Stale Check: Last={last_candle_aware} Now={now_utc} Diff={hours_diff:.2f}h")
            
            # 2. Hard Weekend Block (Forex Logic)
            # Saturday (5) is always CLOSED
            # Sunday (6) is CLOSED until ~21:00 UTC (Sydney Open)
            weekday = now_utc.weekday()
            hour = now_utc.hour
            
            if weekday == 5: # Saturday
                return True
            if weekday == 6 and hour < 21: # Sunday before 5PM EST (approx)
                 # Even if data looks "fresh" (e.g. crypto or glitch), for Forex pairs we block.
                 # Note: This might block Crypto if mixed. Assuming Forex context here.
                 if hours_diff > 1.0: # Double verify it's not actually live data
                     return True

            # 3. Staleness Threshold (Relaxed for Monday mornings due to yfinance lag)
            # If it's Monday and we have Friday's data, allow it until noon UTC
            is_monday_morning = (now_utc.weekday() == 0 and now_utc.hour < 12)
            if is_monday_morning and hours_diff < 72:
                return False

            # For 1h candles, if we are > 4 hours past the close, we are missing a candle.
            if hours_diff > 4.0:
                 return True
                 
            return False
            
        except Exception as e:
            # Identify as stale if check fails to prevent bad signals
            return True

    def _update_global_context(self) -> Dict[str, pd.DataFrame]:
        """Refreshes the Global Market data (all pairs + gold) for CSM computing."""
        now = datetime.now()
        if self._global_data_cache and self._last_global_update and (now - self._last_global_update).total_seconds() < 900:
            return self._global_data_cache
            
        logger.info("Intelligence Context: Refreshing Global Market Matrix...")
        symbols = self.data_engine.get_all_pairs()
        raw_data = {}

        # Context symbols (Gold and Bond Yields)
        context_symbols = ["GOLD", "^TNX"]
        for s in context_symbols:
            try:
                # Use yfinance for context symbols to match training data
                df = self.data_engine.fetch(s, interval="1h", days=7)
                if df is not None and not df.empty:
                    raw_data[s] = df
            except: pass

        for s in symbols:
            try:
                df = self.data_engine.fetch(s, interval="1h", days=5)
                if not df.empty: raw_data[s] = df
            except: continue
        
        common_index = None
        for df in raw_data.values():
            if common_index is None: common_index = df.index
            else: common_index = common_index.intersection(df.index)
            
        aligned = {s: df.reindex(common_index).ffill().bfill() for s, df in raw_data.items()}
        self._global_data_cache = aligned
        self._last_global_update = now
        return aligned

    def load_foundation_model(self, symbol: str) -> Optional[Dict]:
        """Load the Global Foundation TFT model and pair-specific adaptation."""
        base_dir = Path("models/foundation")
        model_path = base_dir / "foundation_brain.keras"
        config_path = base_dir / "config.json"
        
        if not model_path.exists():
            return None
            
        try:
            # For TFT we might need custom objects if implemented as layers
            from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork
            custom_objects = {
                'VariableSelectionNetwork': VariableSelectionNetwork,
                'GatedResidualNetwork': GatedResidualNetwork
            }
            model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
            logger.info(f"✅ Loaded Foundation Model (TFT) for {symbol}")
            
            # Load universal scaler (should be saved in foundation dir)
            scaler_path = base_dir / "scaler.joblib"
            scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else joblib.load(str(PROJECT_ROOT / "models" / "specialist" / symbol / "BUY" / "scaler.joblib"))
            
            # Load trades volume from config.json
            trades_count = 0
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                        trades_count = config_data.get('total_samples', 0)
                except: pass

            return {
                'model': model,
                'scaler': scaler,
                'model_type': 'foundation_tft',
                'model_trades': trades_count
            }
        except Exception as e:
            logger.error(f"Failed to load Foundation Model: {e}")
            return None

    def load_phase3_expert(self, symbol: str) -> Optional[Dict]:
        """Load Phase 3 Transfer-Learned Expert Model."""
        cache_key = f"{symbol}_phase3"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        try:
            expert_path = PROJECT_ROOT / "models" / "expert" / symbol / "expert_model.keras"
            config_path = PROJECT_ROOT / "models" / "expert" / symbol / "config.json"
            scaler_path = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
            
            if not expert_path.exists() or not scaler_path.exists():
                logger.debug(f"Phase 3 expert NOT FOUND for {symbol} at {expert_path}")
                return None
            
            logger.info(f"Loading Phase 3 Expert Adapter for {symbol} from {expert_path}")
            
            model = keras.models.load_model(str(expert_path))
            scaler = joblib.load(str(scaler_path))
            
            trades = 0
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    trades = cfg.get('trades', 0)
                    
            models = {
                'model': model,
                'scaler': scaler,
                'model_type': 'expert_adapted', # Follows Foundation TFT sizing (32 cols)
                'model_trades': trades
            }
            self._model_cache[cache_key] = models
            logger.info(f"Loaded Phase 3 Expert Adapter for {symbol}")
            return models
        except Exception as e:
            # logger.warning(f"Failed to load Phase 3 expert for {symbol}: {e}")
            return None

    def predict_symbol(
        self,
        symbol: str,
        save_to_db: bool = True,
        win_rate: Optional[str] = None,
        allow_stale: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for a specific accuracy tier (Model Isolation Mode).
        Exposes trade volume for the selected model.
        """
        try:
            # ── 0. SIGNAL LOCKING CHECK (STRICT) ──────────────────────────
            # Rule: If a symbol is already ACTIVE (Live or Shadow), we LOCK it.
            # This prevents re-predicting or conflicting signals until resolution.
            if self.db:
                # Include hidden=True to lock on shadow trades too
                active_signals = self.db.get_active_signals(symbol, include_hidden=True)
                real_active = [s for s in active_signals if s.get('signal') in ('BUY', 'SELL')]
                
                if real_active:
                    lock = real_active[0]
                    logger.info(f"🔒 {symbol}: Active Signal detected (ID: {lock['id']}). Locking system state.")
                    return {
                        'id': lock['id'],
                        'timestamp': lock['timestamp'],
                        'symbol': lock['symbol'],
                        'signal': lock['signal'],
                        'confidence': float(lock.get('confidence', 0.0)),
                        'raw_confidence': float(lock.get('raw_confidence', 0.0)),
                        'buy_prob': float(lock.get('buy_prob', 0.0)),
                        'sell_prob': float(lock.get('sell_prob', 0.0)),
                        'wait_prob': float(lock.get('wait_prob', 0.0)),
                        'price_at_signal': float(lock.get('price_at_signal', 0.0)),
                        'tp_price': float(lock.get('tp_price', 0.0)),
                        'sl_price': float(lock.get('sl_price', 0.0)),
                        'tp_pips': int(lock.get('tp_pips', 0)),
                        'sl_pips': int(lock.get('sl_pips', 0)),
                        'winning_tier': lock.get('winning_tier', f"{target_int}%" if 'target_int' in locals() else "60%"),
                        'model_trades': lock.get('model_trades', 0),
                        'regime': lock.get('regime', 'Trending'),
                        'is_locked': True
                    }

            # 1. Determine Target Tier
            target_int = 60
            if win_rate:
                if win_rate == "Apex": target_int = 95
                elif win_rate == "Expert": target_int = 90
                else: 
                    try: target_int = int(win_rate.replace('%', ''))
                    except: target_int = 60
            
            logger.info(f"[PREDICT] PREDICTION REQUEST: {symbol} @ {target_int}% (Input: {win_rate})")
            
            
            df = self.data_engine.fetch(symbol, interval="1h", days=30)  # 30 days for EMA200 warmup
            if df is None or len(df) < 60: return None
            if not allow_stale and self._is_data_stale(df.index[-1]): return None

            # ── Regime Detection Gate (PHASE 4) ─────────────────────────────────────────
            tradeable, dynamic_threshold, regime_result = self._regime_detector.is_tradeable(df, symbol)
            regime_label = regime_result.regime.value if regime_result else "UNKNOWN"
            
            if not tradeable:
                logger.warning(f"⛔ {symbol} BLOCKED by Regime Detector: status={regime_label}")
                return {
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'regime': regime_label,
                    'reason': 'Regime Block'
                }

            # 1. Base Features
            base_features = self.feature_engineer.extract_features(df)
            
            # 2. Intelligence Enrichment (Global Context)
            global_data = self._update_global_context()
            features = self.global_engineer.add_global_features(symbol, base_features, global_data)
            
            # 3. Load Model (Priority 1: Phase 3 Expert, Priority 2: Phase 2 Foundation)
            models = self.load_phase3_expert(symbol)
            if not models:
                models = self.load_foundation_model(symbol)
            
            if not models:
                # Fallback to specialists only if Foundation is missing or fails
                models = self.load_models(symbol, win_rate=target_int)

            if not models:
                logger.warning(f"No model (Foundation or Specialist) found for {symbol} at {target_int}% tier.")
                return None
            
            # Signal Generation
            # If Global Intelligence was added, 'features' already contains it.
            # Only use base_features as a backup for legacy models.
            if models.get('model_type') not in ['foundation_tft', 'expert_adapted']:
                features = base_features.copy()
            
            scaler = models['scaler']
            expected_features = scaler.n_features_in_
            current_features = len(features.columns)
            
            # Feature Adaptation (Skip for Foundation Models as they use the 32-column Global Matrix)
            if models.get('model_type') in ['foundation_tft', 'expert_adapted']:
                pass # Already aligned to 32 features by global_engineer.add_global_features
            elif expected_features == 22 and current_features == 19:
                for i in range(3): features[f'pad_{i}'] = 0.0
            elif expected_features >= 25:
                correlated = self._get_correlated_assets(symbol)
                for asset in correlated:
                    try:
                        asset_df = self.data_engine.fetch(asset['symbol'], interval="1h", days=14)
                        if asset_df is not None:
                            features = self.feature_engineer.add_correlated_asset(features, asset_df, asset_name=asset['symbol'])
                    except: pass
                current_features = len(features.columns)
                if current_features < expected_features:
                    for i in range(expected_features - current_features): features[f'pad_rem_{i}'] = 0.0
            elif expected_features == 20 and current_features == 19:
                features['pad_0'] = 0.0
            elif current_features < expected_features:
                for i in range(expected_features - current_features): features[f'pad_gen_{i}'] = 0.0
            
            # Final sanity check: ensure column order matches scaler expectation for non-TFT
            # (TFT handles this via the fixed 32-column matrix)
            current_features = len(features.columns)
            if models.get('model_type') in ['binary', 'expert']:
                # For binary/expert, check buy_model
                input_shape = models['buy_model'].input_shape
            else:
                input_shape = models['model'].input_shape
            
            # input_shape is (None, sequence_length, features)
            seq_len = input_shape[1] if len(input_shape) > 1 else 60
            logger.info(f"Detected model sequence length: {seq_len}")

            X, _ = self.feature_engineer.create_sequences(features, pd.Series(0, index=features.index), sequence_length=seq_len)
            if len(X) == 0: return None
            
            X_last = X[-1].reshape(1, seq_len, -1)
            if X_last.shape[2] != expected_features:
                if X_last.shape[2] > expected_features: X_last = X_last[:, :, :expected_features]
                else: X_last = np.pad(X_last, ((0,0), (0,0), (0, expected_features - X_last.shape[2])), 'constant')
            
            X_scaled = scaler.transform(X_last.reshape(-1, expected_features)).reshape(1, seq_len, expected_features)
            
            # ── Static Thresholds (Reverted from Regime-Adjusted) ──────────────────────────
            buy_threshold  = models.get('buy_threshold', 0.70)
            sell_threshold = models.get('sell_threshold', 0.70)
            
            # Initialize default probabilities (safe defaults if a direction is blocked)
            buy_prob = 0.0
            sell_prob = 0.0
            wait_prob = 1.0
            signal = "WAIT"
            confidence = 0.0

            if models.get('model_type') in ['binary', 'expert']:
                # Only run prediction on qualified directions
                if models.get('buy_qualified', True) and models.get('buy_model') is not None:
                    buy_prob = float(models['buy_model'].predict(X_scaled, verbose=0)[0][0])
                if models.get('sell_qualified', True) and models.get('sell_model') is not None:
                    sell_prob = float(models['sell_model'].predict(X_scaled, verbose=0)[0][0])

                wait_prob = max(0.0, 1.0 - max(buy_prob, sell_prob))
                
                # ── 1. PROVEN OPPORTUNITY OVERRIDE ────────────────────────────
                # Rule: If proven at 70% accuracy, floor is 60% confidence.
                # We check the performance gate for BOTH directions at their current probabilities.
                buy_proven = (buy_prob >= 0.60) and self.perf_gate.is_tier_approved(symbol, buy_prob)
                sell_proven = (sell_prob >= 0.60) and self.perf_gate.is_tier_approved(symbol, sell_prob)

                # ── 2. Determine Dominant Signal ────────────────────────────
                if (buy_prob >= buy_threshold or buy_proven) and buy_prob > sell_prob:
                    signal, confidence = "BUY", buy_prob
                elif (sell_prob >= sell_threshold or sell_proven) and sell_prob > buy_prob:
                    signal, confidence = "SELL", sell_prob
                else:
                    signal, confidence = "WAIT", wait_prob
                
                # Hard gate for strictly approved pairs
                is_tier_proven = buy_proven if signal == 'BUY' else sell_proven if signal == 'SELL' else False
            else:
                # 3-Class Foundation Model
                proba = models['model'].predict(X_scaled, verbose=0)[0]
                wait_prob, buy_prob, sell_prob = float(proba[0]), float(proba[1]), float(proba[2])
                
                # Check provenness for 3-class logic
                buy_proven = (buy_prob >= 0.60) and self.perf_gate.is_tier_approved(symbol, buy_prob)
                sell_proven = (sell_prob >= 0.60) and self.perf_gate.is_tier_approved(symbol, sell_prob)

                if (buy_prob >= buy_threshold or buy_proven) and buy_prob > sell_prob:
                    signal, confidence = "BUY", buy_prob
                elif (sell_prob >= sell_threshold or sell_proven) and sell_prob > buy_prob:
                    signal, confidence = "SELL", sell_prob
                else:
                    signal, confidence = "WAIT", wait_prob

            # Determine the HIGHEST directional conviction for the UI (Buy or Sell)
            # This ensures the dashboard shows 60% Sell even if the verdict is still WAIT.
            if buy_prob > sell_prob:
                raw_confidence = buy_prob
            elif sell_prob > buy_prob:
                raw_confidence = sell_prob
            else:
                raw_confidence = wait_prob
            
            # Recalculate proven/authorized state for 3-class models
            # Get the status from the performance gate for this confidence level
            tier_status = self.perf_gate.get_tier_status(symbol, raw_confidence)
            is_tier_proven = (tier_status == "APPROVED")
            is_tier_benched = (tier_status == "BENCHED")
                
            # ── Phase 4: Platt Scaling Calibration ───────────────────────────
            # Map raw model confidence to real-world win rate
            try:
                final_confidence = self.calibrator.calibrate(symbol, signal, raw_confidence)
                logger.info(f"⚖️ {symbol} {signal} Calibrated: {raw_confidence:.1%} -> {final_confidence:.1%}")
            except Exception as e:
                logger.warning(f"Calibration failed for {symbol}: {e}")
                final_confidence = raw_confidence

            # ── 5. Strict Tier-Specific Authorization ──────────────────
            # Rule: Only take trades at specialized confidence tiers (60/70/80...)
            # if that specific tier is certified in the performance matrix.
            
            # Determine currently applicable tier (bucket)
            conf_int = int(raw_confidence * 100)
            applicable_tier = 0
            for t in [60, 70, 80, 90, 100]:
                if conf_int >= t:
                    applicable_tier = t
            
            # Check for Tier-Specific Approval (Proven status)
            is_tier_proven = self.perf_gate.is_tier_approved(symbol, raw_confidence)
            static_hurdle = target_int / 100.0
            
            is_authorized = False
            is_hidden = 0
            
            if is_tier_proven:
                # If specific tier is certified, we TRUST the 60% floor.
                logger.info(f"✅ {symbol}: {applicable_tier}% Tier is CERTIFIED. Authorizing LIVE trade.")
                is_authorized = True
                is_hidden = 0
            elif is_tier_benched and (raw_confidence >= 0.60):
                # If benched, authorize a "Silent/Shadow" trade to accumulate history
                logger.info(f"🤫 {symbol}: {applicable_tier}% Tier is BENCHED. Authorizing SHADOW certification trade.")
                is_authorized = True
                is_hidden = 1 # Hide from UI/Telegram
            elif final_confidence >= static_hurdle:
                # If hit the high safety hurdle, we allow it but maybe hide if not certified
                logger.info(f"⚠️ {symbol}: Tier {applicable_tier}% not certified. Hit standard hurdle ({static_hurdle:.1%}).")
                is_authorized = True
                is_hidden = 0 if is_tier_proven else 1 # Only show high-hurdle if proven
            else:
                logger.info(f"⛔ {symbol}: {final_confidence:.1%} < Hurdle {static_hurdle:.1%}. Signal={signal} Benched (Waiting).")
                is_authorized = False
                is_hidden = 1

            expert_signal = signal # Save the model's intended direction for shadow history
            
            if not is_authorized:
                signal = "WAIT"
            elif signal == "WAIT":
                # Promotion: If authorized (Benched/Proven) but currently WAIT due to threshold,
                # adopt the AI's dominant bias for certification/live execution.
                signal = "BUY" if buy_prob > sell_prob else "SELL"
                confidence = buy_prob if signal == "BUY" else sell_prob
                logger.info(f"🚀 {symbol}: Promoting signal from WAIT to {signal} for {applicable_tier}% certification.")

            if signal in ('BUY', 'SELL') and models.get('model_type') not in ('binary', 'expert'):
                # PROVEN OVERRIDE: If the signal is officially authorized by the 60% Proven floor, 
                # we skip the secondary heatmap split test for stability.
                if not is_authorized:
                    winner_pct = p_buy if signal == 'BUY' else p_sell
                    logger.info(f"DEBUG: {symbol} winner_pct={winner_pct:.4f}, dynamic_threshold={dynamic_threshold:.4f}, p_wait={p_wait:.4f}")
                    if winner_pct < dynamic_threshold:
                        logger.info(f"⛔ {symbol}: Heatmap split — winner only {winner_pct:.1%} (need >{dynamic_threshold:.0%}). Downgrading to WAIT.")
                        signal = "WAIT" 
                    elif p_wait > 0.40:
                        logger.info(f"⛔ {symbol}: Heatmap wait too high — {p_wait:.1%} (need <40%). Downgrading to WAIT.")
                        signal = "WAIT" 
                else:
                    logger.info(f"⚡ {symbol}: Bypassing Heatmap Gate (Proven/Hurdle Authorized).")
            
            current_price = float(df['close'].iloc[-1])
            levels = {'tp_price': 0.0, 'sl_price': 0.0, 'tp_pips': 0, 'sl_pips': 0}
            
            # We calculate TP/SL for ANY actionable bias (Live OR Benched)
            # This is required for shadow certification tracking.
            target_signal = signal if signal != "WAIT" else expert_signal
            if target_signal in ("BUY", "SELL"):
                # Pip size: 0.01 for Gold, 0.01 for JPY, 0.0001 for others
                pip_size = 0.01 if ('XAU' in symbol or 'GOLD' in symbol or 'JPY' in symbol) else 0.0001
                atr_pips = (float(features['atr_norm'].iloc[-1]) * current_price) / pip_size
                levels = self.calculate_tp_sl(symbol, target_signal, current_price, atr_pips=atr_pips)

            # Metadata with Volume — use correct direction's trade count
            # 2. Metadata & Volume Attribution
            # Prioritize volume from the loaded model object itself
            trades = models.get('model_trades', 0)
            if trades == 0:
                if signal == 'BUY': trades = models.get('buy_trades', 0)
                elif signal == 'SELL': trades = models.get('sell_trades', 0)
            
            if trades == 0:
                # Fallback to filesystem only if model dict is missing volume data
                try:
                    base = Path("models")
                    # Try specific expertise branch first
                    specific_path = base / symbol / str(target_int) / (signal if signal != "WAIT" else "BUY") / "config.json"
                    if specific_path.exists():
                        with open(specific_path, 'r') as f:
                            c = json.load(f)
                            trades = c.get('trades', 0) or c.get('total_samples', 0)
                    
                    # Try pair root second
                    if trades == 0:
                        alt_conf = base / symbol / "config.json"
                        if alt_conf.exists():
                            with open(alt_conf, 'r') as f:
                                c = json.load(f)
                                trades = c.get('trades', 0) or c.get('total_samples', 0)
                except: pass

            result = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'signal': signal,
                'expert_signal': expert_signal, # NEW: The original bias for Sentinel/Shadow monitor
                'confidence': final_confidence,
                'confidence_tier': applicable_tier, # NEW: Numeric tier for Escalation logic
                'raw_confidence': raw_confidence,
                'buy_prob': float(buy_prob),
                'sell_prob': float(sell_prob),
                'wait_prob': float(wait_prob),
                'price_at_signal': current_price,
                'tp_price': levels['tp_price'],
                'sl_price': levels['sl_price'],
                'tp_pips': levels['tp_pips'],
                'sl_pips': levels['sl_pips'],
                'winning_tier': f"{target_int}%",
                'model_trades': trades,
                'model_version': models.get('model_type', 'foundation_tft'),
                'regime': regime_label,
                'regime_threshold': dynamic_threshold,
                'is_proven': int(is_tier_proven),
                'is_hidden': int(is_hidden),
                'outcome': 'ACTIVE' if (is_authorized and signal != "WAIT") else 'N/A',
                'adx': 0.0,
                'atr_zscore': 0.0,
                'vix_proxy': round(float(features['vix_proxy'].iloc[-1]), 4) if 'vix_proxy' in features.columns else 0.0,
                'yield_slope': round(float(features['yield_curve_slope'].iloc[-1]), 4) if 'yield_curve_slope' in features.columns else 0.0,
            }

            # 2.5 Enrich with REALIZED Win Rates from Database
            realized_perf = self.db.get_symbol_win_rates(symbol)
            result['buy_win_rate'] = realized_perf['buy_win_rate']
            result['sell_win_rate'] = realized_perf['sell_win_rate']
            
            # Use realized win rate for the primary signal if applicable
            if signal == 'BUY':
                result['realized_win_rate'] = realized_perf['buy_win_rate']
            elif signal == 'SELL':
                result['realized_win_rate'] = realized_perf['sell_win_rate']
            else:
                result['realized_win_rate'] = 0.0

            # ── 3. Prop-Grade Lot Calculation ──────────────────────────
            if signal in ('BUY', 'SELL'):
                result['suggested_lots'] = self.calculate_lots_precision(symbol, current_price, levels['sl_price'])
            else:
                result['suggested_lots'] = 0.0

            # 4. Handle Result & Cooldown
            if save_to_db:
                should_save = True
                
                # Check Database for recent signals to make cooldown stateless
                recent_db_signals = self.db.get_recent_signals(limit=5, symbol=symbol)
                
                # ── Correlation Cluster Filter ────────────────────────────
                # Prevent taking trades on highly correlated pairs to limit exposure
                correlated_assets_config = self._get_correlated_assets(symbol)
                highly_correlated_symbols = [
                    a['symbol'] for a in correlated_assets_config 
                    if abs(a.get('correlation', 0.0)) >= 0.70
                ]
                
                if highly_correlated_symbols:
                    all_recent_signals = self.db.get_recent_signals(limit=20)
                    active_correlated = [
                        s for s in all_recent_signals 
                        if s.get('outcome') == 'ACTIVE' 
                        and s.get('signal') in ('BUY', 'SELL')
                        and s.get('symbol') in highly_correlated_symbols
                    ]
                    if active_correlated:
                        logger.info(f"🧱 CORRELATION BLOCK: {symbol} is highly correlated to an active trade ({active_correlated[0]['symbol']}). Blocking new generation to manage risk.")
                        if save_to_db:
                            return None

                # STRICT LOCK: If there is ANY active signal for this symbol, block new SAVING of signals.
                # Only apply lock if the active signal is a real trade (BUY/SELL). Pure WAIT audit signals don't block.
                active_signals = [s for s in recent_db_signals if s.get('outcome') == 'ACTIVE' and s.get('signal') in ('BUY', 'SELL')]
                if active_signals:
                    logger.info(f"🔒 {symbol} has active signal (ID: {active_signals[0]['id']}). Blocking new generation task (UI still shows bias).")
                    if save_to_db:
                        # Only return None for background-saving callers like main.py
                        return None

                for recent in recent_db_signals:
                    if recent['signal'] == signal:
                        try:
                            sig_time = datetime.fromisoformat(recent['timestamp'])
                            if sig_time.tzinfo is None:
                                sig_time = sig_time.replace(tzinfo=timezone.utc)
                            
                            time_since = (datetime.now(timezone.utc) - sig_time).total_seconds() / 60
                            # Bypass cooldown for certified BUY/SELL signals — they must always save
                            # so the locking mechanism can pick them up on the next predict call.
                            # Only WAIT signals respect the deduplication cooldown.
                            if signal in ('BUY', 'SELL') and is_tier_proven:
                                logger.info(f"✅ {symbol}: Certified {signal} — bypassing cooldown ({time_since:.1f} min). Saving.")
                                break  # Force should_save=True
                            elif time_since < 60:
                                logger.info(f"⏳ {symbol} {signal} on cooldown. Last signal {time_since:.1f} minutes ago.")
                                should_save = False
                                break
                        except Exception:
                            pass
                
                if should_save:
                    logger.info(f"🎯 SAVING ISOLATED SIGNAL: {symbol} {signal} from {target_int}% Expert (Vol: {trades})")
                    self.db.save_signal(result)
                    
                    # Ensure telegram fires natively if we bypass the executive loop
                    # ONLY send Telegram alerts for Certified (Proven) signals
                    if result.get('signal') in ('BUY', 'SELL') and is_tier_proven:
                        self.notifier.send_signal_alert(result)
                        
                    return result
                else:
                    # Return None so callers (Executive/main) don't send duplicate alerts
                    return None
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {symbol} @ {win_rate}: {e}", exc_info=True)
            return None

    def run_all(self, win_rate: Optional[str] = None) -> List[Dict]:
        """Run inference on all symbols and return results."""
        symbols = self.data_engine.get_all_pairs()
        logger.info(f"Running bulk inference for {len(symbols)} pairs...")
        
        results = []
        for symbol in symbols:
            try:
                result = self.predict_symbol(symbol, win_rate=win_rate, save_to_db=True)
                if result:
                    results.append(result)
                
                # Pacing delay for memory cleanup and CPU stability
                time.sleep(1)
            except Exception as e:
                logger.error(f"Bulk inference failed for {symbol}: {e}")
                
        return results

    def run_continuous(
        self,
        interval_minutes: int = 5,
        symbols: Optional[List[str]] = None
    ):
        """
        Run continuous monitoring loop.
        
        Args:
            interval_minutes: Minutes between scans
            symbols: List of symbols to monitor (None = all configured pairs)
        """
        if symbols is None:
            symbols = self.data_engine.get_all_pairs()
        
        logger.info(f"Starting continuous inference for {len(symbols)} pairs")
        logger.info(f"Scan interval: {interval_minutes} minutes")
        logger.info(f"Monitoring: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        
        try:
            while True:
                start_time = time.time()
                
                logger.info(f"--- Scan started at {datetime.now().strftime('%H:%M:%S')} ---")
                
                signals_generated = 0
                for symbol in symbols:
                    result = self.predict_symbol(symbol, save_to_db=True)
                    if result:
                        signals_generated += 1
                    
                    # Pacing delay for memory cleanup and CPU stability
                    time.sleep(1)
                
                elapsed = time.time() - start_time
                logger.info(
                    f"--- Scan complete: {signals_generated} signals in {elapsed:.1f}s ---"
                )
                
                # Sleep until next interval
                sleep_time = max(1, interval_minutes * 60 - elapsed)
                logger.info(f"Next scan in {sleep_time/60:.1f} minutes\n")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Inference engine stopped by user")
        except Exception as e:
            logger.error(f"Inference engine crashed: {e}", exc_info=True)
            raise


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    engine = InferenceEngine()
    
    # Test single prediction
    result = engine.predict_symbol("EURUSD", win_rate="90%", save_to_db=False, allow_stale=True)
    if result:
        print(f"\n{result['signal']} {result['symbol']} @ {result['price_at_signal']:.5f}")
        print(f"Tier: {result['winning_tier']}, Volume: {result['model_trades']} Trades")
        print(f"TP: {result['tp_price']:.5f} (+{result['tp_pips']} pips)")
        print(f"SL: {result['sl_price']:.5f} (-{result['sl_pips']} pips)")
    else:
        print("No signal detected")
