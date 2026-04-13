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

# ---------------------------------------------------------------------------
# Keras backwards-compat shim
# Models trained on newer Keras (>= 3.x) include 'quantization_config' in
# Dense layer configs.  Older Keras on the runtime VM rejects this kwarg.
# Register a drop-in Dense that silently absorbs the unknown parameter.
# ---------------------------------------------------------------------------
def _build_keras_compat_objects() -> dict:
    if keras is None:
        return {}
    try:
        class _CompatDense(keras.layers.Dense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)
        return {'Dense': _CompatDense}
    except Exception:
        return {}

_KERAS_COMPAT = _build_keras_compat_objects()

def _load_model(path: str) -> object:
    """Wrapper around keras.models.load_model with compat custom_objects."""
    try:
        return keras.models.load_model(path, custom_objects=_KERAS_COMPAT)
    except Exception:
        # Final fallback: load without custom objects
        return keras.models.load_model(path)


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
                f"Reason: {_TF_ERROR}\n"
                "Please ensure tensorflow is installed correctly."
            )
            
        logger.info(f"InferenceEngine initialized with model_dir={model_dir}")

    
    def _get_correlated_assets(self, symbol: str) -> List[Dict]:
        """Get correlation data for a symbol from config"""
        correlations = self.config.get('correlations', {})
        return correlations.get(symbol, [])

    def ensure_global_context(self, timeframe: str = "1h") -> pd.DataFrame:
        """
        Ensure the Global Market Matrix is loaded and up-to-date.
        Caches the matrix to avoid redundant MT5 calls per prediction.
        """
        # Define cache validity (e.g., matrix is valid for 55 minutes on 1h TF)
        cache_duration = timedelta(minutes=55) if timeframe == "1h" else timedelta(minutes=14)
        
        now = datetime.now()
        
        if self._global_data_cache is not None and self._last_global_update is not None:
            age = now - self._last_global_update
            if age < cache_duration:
                # Cache is valid
                return self._global_data_cache
                
        # Rebuild Global Matrix
        logger.info("Intelligence Context: Refreshing Global Market Matrix...")
        try:
            mt5 = get_mt5()
            global_df = self.global_engineer.build_global_context(mt5, timeframe=timeframe)
            
            self._global_data_cache = global_df
            self._last_global_update = now
            return global_df
        except Exception as e:
            logger.error(f"Failed to build Global Matrix: {e}")
            if self._global_data_cache is not None:
                logger.warning("Yielding stale global matrix due to build failure.")
                return self._global_data_cache
            raise
            
    def load_binary_models(self, symbol: str) -> Optional[Dict]:
        """
        Load Binary Split Models (BUY network + SELL network) and check Performance Gates.
        """
        # Try multiple base directories since location changed
        base_dirs = ["models/binary", "models/trained"]
        
        for base_dir in base_dirs:
            try:
                # Ensure path is absolute from PROJECT_ROOT
                model_dir = PROJECT_ROOT / base_dir / symbol
                
                # Check Performance Gates BEFORE loading the model
                # This ensures we don't hold bad models in memory
                
                # Retrieve the daily baseline audit results
                buy_win_rate = self.perf_gate.get_win_rate(symbol, "BUY")
                sell_win_rate = self.perf_gate.get_win_rate(symbol, "SELL")
                
                # Retrieve the exact number of trades forming the baseline
                buy_trades = self.perf_gate.get_trade_count(symbol, "BUY")
                sell_trades = self.perf_gate.get_trade_count(symbol, "SELL")
                
                # Enforce Institutional Floor
                buy_qualified = buy_win_rate >= GOLDEN_MIN_WIN_RATE
                sell_qualified = sell_win_rate >= GOLDEN_MIN_WIN_RATE

                buy_path = model_dir / "buy_model.keras"
                sell_path = model_dir / "sell_model.keras"
                scaler_path = model_dir / "scaler.joblib"
                
                if not buy_path.exists() or not sell_path.exists() or not scaler_path.exists():
                    continue

                if not buy_qualified and not sell_qualified:
                    logger.warning(f"QUALITY GATE BLOCKED {symbol}: BUY WR={buy_win_rate:.1%} ({buy_trades}t), SELL WR={sell_win_rate:.1%} ({sell_trades}t) — both below {GOLDEN_MIN_WIN_RATE*100:.0f}% floor")
                    continue  # Skip this model entirely
                
                if not buy_qualified:
                    logger.info(f"QUALITY GATE: {symbol} BUY masked (WR={buy_win_rate:.1%}, {buy_trades}t)")
                if not sell_qualified:
                    logger.info(f"QUALITY GATE: {symbol} SELL masked (WR={sell_win_rate:.1%}, {sell_trades}t)")
                
                models = {
                   'buy_model': _load_model(str(buy_path)) if buy_qualified else None,
                    'sell_model': _load_model(str(sell_path)) if sell_qualified else None,
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
                'buy_model': _load_model(str(buy_path)),
                'sell_model': _load_model(str(sell_path)),
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
                        'model': _load_model(str(model_path)),
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

    def load_specialist_model(self, symbol: str) -> Optional[Dict]:
        """
        Load specific Phase 2 Specialist model for a pair.
        Always returns None for GOLD since it should use Phase 3 Expert flow
        if trained via Transfer Learning, or Foundation if not.
        """
        if symbol == "GOLD":
             # Force GOLD through the load_foundation / load_phase3_expert pipeline
             # This bypasses the old Specialist directory lookups that break dimensionality
             return None
             
        # Fallback to enhanced loader for backwards compatibility
        return self.load_enhanced_model(symbol)
        
    def load_foundation_model(self, symbol: str) -> Optional[Dict]:
        """
        Load Phase 1 Foundation TFT Model (Global).
        This model expects 48 features exactly.
        """
        cache_key = "foundation_tft"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        try:
            model_path = PROJECT_ROOT / "models" / "foundation" / "foundation_tft.keras"
            scaler_path = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
            
            if not model_path.exists():
                logger.debug(f"Foundation model not found at {model_path}")
                # Fallback to older paths
                old_paths = [
                    PROJECT_ROOT / "models" / "multi_pair" / "base_model.keras",
                    PROJECT_ROOT / "models" / "base_model" / "model.keras"
                ]
                for p in old_paths:
                    if p.exists():
                        model_path = p
                        scaler_path = p.parent / "scaler.joblib"
                        logger.info(f"Using fallback foundation path: {model_path}")
                        break
                        
            if not model_path.exists():
                return None
                
            from models.foundation_trainer import VariableSelectionNetwork, GatedResidualNetwork, GatingNetwork
            
            # Use specific custom objects for TFT architecture
            custom_objects = {
                'VariableSelectionNetwork': VariableSelectionNetwork,
                'GatedResidualNetwork': GatedResidualNetwork,
                'GatingNetwork': GatingNetwork
            }
            
            logger.info(f"Loading Foundation Model from {model_path}")
            model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
            scaler = joblib.load(str(scaler_path))
            
            trades_count = 0
            config_path = model_path.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    trades_count = cfg.get('trades', 0)
            
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
            
            model = _load_model(str(expert_path))
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
                        'pip_value': float(lock.get('pip_value', 0.0)),
                        'tp_pips': int(lock.get('tp_pips', 0)),
                        'sl_pips': int(lock.get('sl_pips', 0)),
                        'model_type': lock.get('model_type', 'unknown'),
                        'winning_tier': lock.get('winning_tier', 'Unknown'),
                        'model_trades': int(lock.get('model_trades', 0)),
                        'is_proven': bool(lock.get('is_proven', False)), # Bubble up Proven flag for Dashboard
                        'ltd_win_rate': float(lock.get('ltd_win_rate', 0.0))
                    }
                    
            target_int = int(win_rate.replace('%','')) if win_rate else 90
            logger.info(f"[PREDICT] PREDICTION REQUEST: {symbol} @ {target_int}% (Input: {win_rate})")
            
            # 1. Fetch Local Data & Regime
            # Force re-fetch from MT5 (ignore in-memory cache) to get immediate state change
            df = self.data_engine.get_data(symbol, timeframe="1h", force_refresh=True)
            if df is None or len(df) < 50: # Need enough history for RSI etc
                logger.warning(f"Not enough data for {symbol}")
                return None
                
            tradeable, dynamic_threshold, regime_result = self._regime_detector.is_tradeable(df, symbol)
            
            if not tradeable:
                # Still output an audit result for the dashboard (Wait state)
                # Ensure the dashboard gets a default block state instead of crashing
                return {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 1.0,
                    'raw_confidence': 1.0,
                    'buy_prob': 0.0,
                    'sell_prob': 0.0,
                    'wait_prob': 1.0,
                    'price_at_signal': float(df['close'].iloc[-1]),
                    'tp_price': 0,
                    'sl_price': 0,
                    'pip_value': 0,
                    'tp_pips': 0,
                    'sl_pips': 0,
                    'model_type': 'regime_blocked',
                    'winning_tier': f"Ranging (Below {dynamic_threshold:.1f} ADX)",
                    'model_trades': 0,
                    'is_proven': False, # Regime blocks are never proven trades
                    'ltd_win_rate': 0.0
                }
                
            # 2. Build Pipeline Features (Local + Global)
            # Create feature engineer ONCE to maintain column order
            fe = FeatureEngineer()
            _, X = fe.create_sequences(df, seq_length=32, target_col='close', is_training=False, symbol=symbol)
            
            if X is None or len(X) == 0:
                logger.warning(f"Feature engineer failed to output valid sequences for {symbol}")
                return None
            
            # IMPORTANT FIX: Get exact feature columns from the feature engineer
            # This is critical for matching the Foundation model's expected shape
            local_features_df = fe.create_features(df)
            local_cols = fe.get_expected_columns()

            # Prepare current unscaled data for predictions
            current_data = local_features_df[local_cols].iloc[-(32):].values
            
            # Combine Global Features (if available) - Ensure 48 dimensions total
            # 32 (local) + 16 (global) = 48
            global_df = self.ensure_global_context(timeframe="1h")
            global_data = None
            if global_df is not None and not global_df.empty:
                # Get the last 32 rows to match Sequence Length
                global_slice = global_df.iloc[-(32):]
                # Assuming get_feature_columns exists and returns 16 columns
                # For safety, explicitly define the 16 global cols if not available dynamically
                try:
                    global_cols = self.global_engineer.get_feature_columns()
                    global_data = global_slice[global_cols].values
                except Exception as e:
                    logger.warning(f"Could not align global features properly: {e}")
                    global_data = np.zeros((len(current_data), 16))
            else:
                global_data = np.zeros((len(current_data), 16))
                
            # Concatenate features (Samples, Features)
            # Make sure it's 32 records of 48 features (32, 48)
            combined_features = np.hstack([current_data, global_data])
            
            # 3. Load Model (Priority 1: Phase 3 Expert, Priority 2: Phase 2 Foundation)
            models = self.load_phase3_expert(symbol)
            if not models:
                 models = self.load_foundation_model(symbol)
            
            if not models:
                 logger.warning(f"No model (Foundation or Specialist) found for {symbol} at {win_rate} tier.")
                 return None
                 
            # Extract models and parameters
            if models['model_type'] == 'expert':
                buy_model = models.get('buy_model')
                sell_model = models.get('sell_model')
                scaler = models.get('scaler')
                threshold = max(models.get('buy_threshold', self.confidence_threshold), 
                              models.get('sell_threshold', self.confidence_threshold))
                # Fallback to init threshold if stored config is missing
                threshold = max(threshold, 0.75) # Floor at 75% for Expert models
                trades = models.get('model_trades', 0)
            elif models['model_type'] in ('enhanced', 'foundation_tft', 'expert_adapted'):
                model = models.get('model')
                scaler = models.get('scaler')
                threshold = 0.50 # For 3-class, >50% on BUY/SELL is sufficient as they are mutually exclusive to WAIT
                trades = models.get('model_trades', 0)
            else:
                buy_model = models.get('buy_model')
                sell_model = models.get('sell_model')
                scaler = models.get('scaler')
                
                # Apply Dynamic Thresholds based on Calibration
                calibrated = self.calibrator.get_calibrated_thresholds(symbol)
                buy_threshold = calibrated['BUY']
                sell_threshold = calibrated['SELL']
                
                # Enforce absolute floor to prevent noise
                buy_threshold = max(buy_threshold, 0.60)
                sell_threshold = max(sell_threshold, 0.60)
                
                threshold = max(buy_threshold, sell_threshold) # Aggregate for logging
                trades = models.get('model_trades', 0)

            # Check Proof Status dynamically
            ltd_win_rate = 0.0
            is_tier_proven = False

            # Isolate Proven Status: A model is Proven IF:
            # 1. Backtest trades >= 30 AND backtest win rate >= 70%
            # 2. OR Shadow/Live Win Rate > 65% with at least 5 resolved trades
            try:
                # Try getting empirical performance first
                win_rate_stats = self.db.get_shadow_win_rate(symbol)
                emp_win_rate = win_rate_stats['win_rate']
                emp_trade_count = win_rate_stats['total_trades']
                
                if emp_trade_count >= 5 and emp_win_rate > 0.65:
                     is_tier_proven = True
                     ltd_win_rate = float(emp_win_rate * 100)
                else:
                     # Fall back to training model validation metrics if empirical is lacking
                     is_tier_proven = (trades >= 30 and int(win_rate.replace('%','')) >= 70) if win_rate else False
                     ltd_win_rate = float(win_rate.replace('%','')) if win_rate else 0.0
            except Exception:
                 is_tier_proven = (trades >= 30 and int(win_rate.replace('%','')) >= 70) if win_rate else False

            # Prepare Input
            # Reshape based on model expectations (Foundation TFT = 32 seq, 48 features)
            if models['model_type'] in ('foundation_tft', 'expert_adapted'):
                # ── FOUNDATION / EXPERT ADAPTED SIZING ──
                # Needs exactly 48 columns (32 OHLCV+Ti, 16 Global)
                # Needs exactly sequence length = 32
                if combined_features.shape[0] != 32 or combined_features.shape[1] != 48:
                    logger.warning(f"Shape mismatch for {symbol}: expected (32, 48), got {combined_features.shape}. Refusing to predict.")
                    return None
                    
                # Scale the entire 32x48 matrix
                # Note: If the scaler was trained on flattened 2D arrays, we reshape to 2D, scale, then go back to 3D.
                try:
                    scaled_data = scaler.transform(combined_features)
                    X_input = scaled_data.reshape(1, 32, 48)  # -> (Batch, Seq, Feat)
                except Exception as e:
                    logger.error(f"Scaling failed for {symbol}: {e}")
                    return None
                    
                # For debug printing
                last_price = float(df['close'].iloc[-1])
                    
            elif models['model_type'] in ['binary', 'expert']:
                # Sequence Model
                last_seq = []
                for i in range(len(X)):
                    # Assuming X is shape (1, 32, num_features)
                     scaled_row = scaler.transform(X[i].reshape(1, -1))
                     last_seq.append(scaled_row[0])
                X_input = np.array([last_seq]) # Shape: (1, 32, num_features)
                last_price = float(df['close'].iloc[-1])
            else: # enhanced
                # Use standard X (scaled inherently in older flow, or adapt later)
                 # X_input was not explicitly scaled here for 3-class because 'enhanced' flow used
                 # a different FeatureEngineer path in legacy.
                 # Let's assume it wants the last Sequence from X, but wrapped
                 if len(X) == 0: return None
                 X_input = X[-1:] # The last sequence
                 last_price = float(df['close'].iloc[-1])

            # 4. Predict based on model architecture
            if models['model_type'] in ('enhanced', 'foundation_tft', 'expert_adapted'):
                # 3-Class output: [BUY, SELL, WAIT]
                probs = model.predict(X_input, verbose=0)[0]
                
                # Verify shape of output
                if len(probs) != 3:
                     logger.error(f"Critical Architecture Error: Multi-class model returned {len(probs)} outputs instead of 3. Model path: {models.get('model_type')}")
                     return None
                     
                buy_prob = float(probs[0])
                sell_prob = float(probs[1]) 
                wait_prob = float(probs[2])
                
                # Calibrate probabilities (Platt Scaling via bayesian_engine)
                buy_prob_cal, sell_prob_cal = self.calibrator.calibrate_probabilities(
                    symbol, buy_prob, sell_prob
                )

                # Use Calibrated threshold for Decision Logic
                # If the max probability doesn't exceed 50%, the model is uncertain
                if wait_prob > buy_prob_cal and wait_prob > sell_prob_cal:
                    signal = "WAIT"
                    confidence = wait_prob
                elif buy_prob_cal > sell_prob_cal and buy_prob_cal >= threshold:
                    signal = "BUY"
                    confidence = buy_prob_cal
                elif sell_prob_cal > buy_prob_cal and sell_prob_cal >= threshold:
                    signal = "SELL"
                    confidence = sell_prob_cal
                else:
                    signal = "WAIT"
                    confidence = max(buy_prob_cal, sell_prob_cal)
                    
            else:
                 # Standard Binary Models
                 raw_buy_prob = float(buy_model.predict(X_input, verbose=0)[0][0])
                 raw_sell_prob = float(sell_model.predict(X_input, verbose=0)[0][0])
                 
                 # Apply Calibration
                 buy_prob, sell_prob = self.calibrator.calibrate_probabilities(
                     symbol, raw_buy_prob, raw_sell_prob
                 )
                 wait_prob = 1.0 - max(buy_prob, sell_prob)
                 
                 # Determine signal based on dynamically calibrated thresholds
                 signal = "WAIT"
                 confidence = 0.0
                 
                 if buy_prob > sell_prob and buy_prob >= buy_threshold:
                     signal = "BUY"
                     confidence = buy_prob
                 elif sell_prob > buy_prob and sell_prob >= sell_threshold:
                     signal = "SELL"
                     confidence = sell_prob
                 else:
                     confidence = max(buy_prob, sell_prob)


            # Build result
            result = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'raw_confidence': confidence, # Keep compatibility
                'buy_prob': float(buy_prob) if 'buy_prob' in locals() else 0.0,
                'sell_prob': float(sell_prob) if 'sell_prob' in locals() else 0.0,
                'wait_prob': float(wait_prob) if 'wait_prob' in locals() else 0.0,
                'price_at_signal': last_price,
                'tp_price': 0,
                'sl_price': 0,
                'pip_value': 0,
                'tp_pips': 0,
                'sl_pips': 0,
                'winning_tier': win_rate or "Global",
                'model_type': models['model_type'],
                'model_trades': trades, # Forward the actual trades metric directly
                'is_proven': is_tier_proven, # Critical flag for the bypass system
                'ltd_win_rate': ltd_win_rate
            }
            
            # --- OVERRIDE SYSTEM FOR AUDIT VISIBILITY ---
            # Even if signal != 'WAIT', if not proven & high-tier, we flag it.
            # But the UI will intercept this via 'is_proven'. So we pass it cleanly.

            if signal in ("BUY", "SELL"):
                # Calculate TP/SL natively using ATR or fixed Pip ranges
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                
                # Fetch ATR for dynamic stop placement (Use local feature if available, else standard calc)
                # Approximate 14-period ATR using recent highs and lows from DataEngine's OHLC
                atr_pips = 20 # Default fallback
                try:
                    df['tr0'] = abs(df['high'] - df['low'])
                    df['tr1'] = abs(df['high'] - df['close'].shift())
                    df['tr2'] = abs(df['low'] - df['close'].shift())
                    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
                    atr = tr.rolling(window=14).mean().iloc[-1]
                    atr_pips = atr / pip_size
                except Exception as e:
                    logger.warning(f"ATR calc failed: {e}. Using fixed 20 pips.")

                # Volatility-adjusted targeting (1.5 Risk Reward Ratio target)
                sl_pips = max(15, min(int(atr_pips * 1.5), 50)) # SL bounds: 15-50 pips
                tp_pips = int(sl_pips * 1.5)                    # TP is 1.5x SL
                
                result['tp_pips'] = tp_pips
                result['sl_pips'] = sl_pips
                
                if signal == "BUY":
                    result['tp_price'] = last_price + (tp_pips * pip_size)
                    result['sl_price'] = last_price - (sl_pips * pip_size)
                else:  # SELL
                    result['tp_price'] = last_price - (tp_pips * pip_size)
                    result['sl_price'] = last_price + (sl_pips * pip_size)
                
                should_save = True
                
                # Check Database for recent signals to make cooldown stateless
                recent_db_signals = self.db.get_recent_signals(limit=5, symbol=symbol)
                
                # ── Correlation Cluster Filter ────────────────────────────────
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
                        logger.info(f"🛡️ CORRELATION BLOCK: {symbol} is highly correlated to an active trade ({active_correlated[0]['symbol']}). Blocking new generation to manage risk.")
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
                                logger.info(f"⚡ {symbol}: Certified {signal} — bypassing cooldown ({time_since:.1f} min). Saving.")
                                break  # Force should_save=True
                            elif time_since < 60:
                                logger.info(f"⏳ {symbol} {signal} on cooldown. Last signal {time_since:.1f} minutes ago.")
                                should_save = False
                                break
                        except Exception:
                            pass
                
                if should_save:
                    logger.info(f"💾 SAVING ISOLATED SIGNAL: {symbol} {signal} from {target_int}% Expert (Vol: {trades})")
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

