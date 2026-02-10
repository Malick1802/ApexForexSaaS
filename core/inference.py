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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from core.database import SignalDatabase
from tensorflow import keras
import joblib


logger = logging.getLogger(__name__)


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
        confidence_threshold: float = 0.65,
        cooldown_minutes: int = 60
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing trained models
            config_path: Path to config.yaml
            confidence_threshold: Minimum confidence for signal generation
            cooldown_minutes: Anti-spam cooldown in minutes (default: 60)
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
        self.db = SignalDatabase()
        
        # Cache for loaded models
        self._model_cache: Dict[str, Dict] = {}
        
        # Recent signals tracker (for deduplication)
        self._recent_signals: Dict[str, datetime] = {}
        self._signal_cooldown_minutes = 60  # Don't repeat same signal within 1 hour
        
        logger.info(f"InferenceEngine initialized with model_dir={model_dir}")
        
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
            return self._model_cache[symbol]
        
        # Try multiple base directories (specialist first for certified models)
        base_dirs = ["models/specialist", self.model_dir]
        
        for base_dir in base_dirs:
            try:
                buy_path = Path(base_dir) / symbol / "BUY" / "model.keras"
                sell_path = Path(base_dir) / symbol / "SELL" / "model.keras"
                scaler_path = Path(base_dir) / symbol / "BUY" / "scaler.joblib"
                
                if not (buy_path.exists() and sell_path.exists() and scaler_path.exists()):
                    continue
                
                models = {
                   'buy_model': keras.models.load_model(str(buy_path)),
                    'sell_model': keras.models.load_model(str(sell_path)),
                    'scaler': joblib.load(str(scaler_path)),
                    'model_type': 'binary'
                }
                
                self._model_cache[symbol] = models
                logger.info(f"Loaded binary models for {symbol} from {base_dir}")
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
            return self._model_cache[cache_key]
            
        try:
            # We look for separate BUY/SELL models in the win_rate folder
            base_dir = self.model_dir if Path(self.model_dir).parts[-1] == str(win_rate) else f"models/{symbol}/{win_rate}"
            base_path = Path(base_dir)
            
            # Check if this path exists relative to root or absolute
            if not base_path.exists():
                # Try relative to project root
                base_path = Path("models") / symbol / str(win_rate)
            
            buy_path = base_path / "BUY" / "model.keras"
            sell_path = base_path / "SELL" / "model.keras"
            
            # Scaler usually in one of them or both
            scaler_path = base_path / "BUY" / "scaler.joblib"
            
            # Configs for thresholds
            buy_config_path = base_path / "BUY" / "config.json"
            sell_config_path = base_path / "SELL" / "config.json"
            
            if not (buy_path.exists() and sell_path.exists()):
                logger.debug(f"Expert models not found for {symbol} @ {win_rate}%")
                return None
                
            # Load Configs to get thresholds
            buy_threshold = 0.5
            sell_threshold = 0.5
            
            if buy_config_path.exists():
                with open(buy_config_path, 'r') as f:
                    buy_threshold = json.load(f).get('threshold', 0.5)
            
            if sell_config_path.exists():
                with open(sell_config_path, 'r') as f:
                    sell_threshold = json.load(f).get('threshold', 0.5)

            models = {
                'buy_model': keras.models.load_model(str(buy_path)),
                'sell_model': keras.models.load_model(str(sell_path)),
                'scaler': joblib.load(str(scaler_path)),
                'model_type': 'expert', # Treated like binary but with custom thresholds
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold
            }
            
            self._model_cache[cache_key] = models
            logger.info(f"Loaded EXPERT models for {symbol} (Target: {win_rate}%)")
            return models
            
        except Exception as e:
            logger.error(f"Failed to load expert model {symbol}/{win_rate}: {e}")
            return None
    
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
                    models = {
                        'model': keras.models.load_model(str(model_path)),
                        'scaler': joblib.load(str(scaler_path)),
                        'model_type': 'enhanced'
                    }
                    
                    logger.info(f"Loaded 3-class model for {symbol} from {base_dir}")
                    return models
                    
                except Exception as e:
                    logger.error(f"Failed to load model for {symbol} from {base_dir}: {e}")
                    continue
        
        return None

        return None

    def load_models(self, symbol: str, win_rate: Optional[int] = None) -> Optional[Dict]:
        """
        Unified model loader.
        Priority:
        1. Expert Models (if win_rate specified)
        2. Enhanced 3-class models
        3. Binary models
        """
        # 1. Expert Model
        if win_rate:
            expert = self.load_expert_model(symbol, win_rate)
            if expert:
                return expert
            logger.warning(f"Requested win rate {win_rate} not found for {symbol}, falling back to default.")

        # 2. Enhanced First
        enhanced = self.load_enhanced_model(symbol)
        """
        Unified model loader.
        Priority:
        1. Enhanced 3-class models (models/enhanced or models/specialist)
        2. Binary models (models/binary or models/specialist)
        """
        # Try enhanced first
        enhanced = self.load_enhanced_model(symbol)
        if enhanced:
            return enhanced
            
        # Fallback to binary
        binary = self.load_binary_models(symbol)
        if binary:
            return binary
            
        return None
    
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
        
        # Default/Static values
        min_sl_pips = trading_config.get('stop_loss_pips', 40)
        
        # Determine pip type
        pip_type = 'jpy' if 'JPY' in symbol else 'standard'
        pip_values = trading_config.get('pip_values', {})
        pip_value = pip_values.get(pip_type, 0.0001)
        
        spread_dist = spread_pips * pip_value
        
        # Rule 1: SL = max(ATR * 1.5, min_sl_pips) + spread
        if atr_pips is not None and atr_pips > 0:
            dynamic_sl = atr_pips * 1.5
            base_sl = max(min_sl_pips, dynamic_sl)
        else:
            base_sl = min_sl_pips
            
        sl_pips = base_sl + spread_pips
        
        # Rule 2: TP = (base_sl * 2) + spread
        tp_pips = (base_sl * 2.0) + spread_pips
        
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

            # For 1h candles, if we are > 2 hours past the close, we are missing a candle.
            if hours_diff > 2.0:
                 return True
                 
            return False
            
        except Exception as e:
            logger.warning(f"Stale check failed: {e}")
            # Identify as stale if check fails to prevent bad signals
            return True
            
    def predict_symbol(
        self,
        symbol: str,
        save_to_db: bool = True,
        win_rate: Optional[str] = None,
        allow_stale: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for a single symbol.
        """
        try:
            # Check cooldown (anti-spam)
            cooldown_key = symbol
            if cooldown_key in self._recent_signals:
                time_since = (datetime.now() - self._recent_signals[cooldown_key]).seconds / 60
                if time_since < self._signal_cooldown_minutes:
                    return None
            
            # Load models
            # Parse win_rate if provided (e.g. "90" or "Apex")
            target_rate = None
            if win_rate:
                if win_rate == "Apex":
                    target_rate = 95
                else:
                    try:
                        target_rate = int(win_rate.replace('%', ''))
                    except:
                        target_rate = None
            
            models = self.load_models(symbol, win_rate=target_rate)
            if not models:
                logger.debug(f"No models available for {symbol}")
                return None
            
            # Fetch latest data
            df = self.data_engine.fetch(symbol, interval="1h", days=5)
            if df is None or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Check for Stale Data (Market Closed)
            last_candle_time = df.index[-1]
            if not allow_stale and self._is_data_stale(last_candle_time):
                # logger.debug(f"Skipping {symbol}: Market closed (Last: {last_candle_time})")
                return None

            # Extract features
            features = self.feature_engineer.extract_features(df)
            
            # Get expected features from scaler
            scaler = models['scaler']
            expected_features = scaler.n_features_in_
            current_features = len(features.columns)
            
            # Dynamic Feature Adaptation
            # Case 1: Enhanced Model (Expects 22 = 19 base + 3 MTF)
            if expected_features == 22 and current_features == 19:
                # Pad 3 zeros for MTF features
                for i in range(3):
                    features[f'pad_{i}'] = 0.0
                    
            # Case 2: Certified with Correlated (Expects 25 = 19 base + 6 correlated)
            elif expected_features >= 25:
                # Add correlated assets
                correlated = self._get_correlated_assets(symbol)
                for asset in correlated:
                    asset_symbol = asset['symbol']
                    try:
                        asset_df = self.data_engine.fetch(asset_symbol, interval="1h", days=5)
                        if asset_df is not None and not asset_df.empty:
                            features = self.feature_engineer.add_correlated_asset(
                                features, asset_df, asset_name=asset_symbol
                            )
                    except Exception:
                        pass
                
                # Re-check count
                current_features = len(features.columns)
                if current_features < expected_features:
                    # Pad remaining if still short
                    diff = expected_features - current_features
                    for i in range(diff):
                        features[f'pad_rem_{i}'] = 0.0
            
            # Case 3: Binary Model (Expects 20 = 19 base + 1?)
            elif expected_features == 20 and current_features == 19:
                # Pad 1 zero
                features['pad_0'] = 0.0
                
            # Fallback: Just pad whatever is missing
            elif current_features < expected_features:
                 diff = expected_features - current_features
                 logger.warning(f"{symbol}: Generic padding {diff} features.")
                 for i in range(diff):
                     features[f'pad_gen_{i}'] = 0.0
            
            # Determine sequence length from model
            seq_len = 60 # Default
            try:
                if 'model' in models:
                     # Keras input shape is (None, seq_len, features)
                     input_shape = models['model'].input_shape
                     if input_shape and len(input_shape) >= 2:
                         seq_len = input_shape[1]
                elif 'buy_model' in models:
                     input_shape = models['buy_model'].input_shape
                     if input_shape and len(input_shape) >= 2:
                         seq_len = input_shape[1]
            except Exception:
                pass # Fallback to 60

            # Ensure we have enough data
            if len(features) < seq_len:
                logger.warning(f"Insufficient features for {symbol}: {len(features)} < {seq_len}")
                return None
            
            # Create sequence
            X, _ = self.feature_engineer.create_sequences(
                features,
                pd.Series(0, index=features.index),
                sequence_length=seq_len
            )
            
            if len(X) == 0:
                return None
            
            # Get last sequence
            X_last = X[-1].reshape(1, seq_len, -1)
            n_features = X_last.shape[2]
            
            # Final check
            if n_features != expected_features:
                logger.warning(f"{symbol}: mismatch after adaptation. Got {n_features}, expected {expected_features}. Trimming/Padding.")
                # Force fit
                if n_features > expected_features:
                    X_last = X_last[:, :, :expected_features]
                elif n_features < expected_features:
                    diff = expected_features - n_features
                    X_last = np.pad(X_last, ((0,0), (0,0), (0, diff)), 'constant')
                n_features = expected_features
            
            X_flat = X_last.reshape(-1, n_features)
            X_scaled = scaler.transform(X_flat).reshape(1, seq_len, n_features)
            
            # Predict based on model type
            model_type = models.get('model_type', 'binary')
            
            # Get thresholds (Default to config, override if Expert model)
            buy_threshold = self.confidence_threshold
            sell_threshold = self.confidence_threshold
            
            # If target_rate (from win_rate) is set, use it as the threshold override
            # This ensures "Apex" (95) acts as a filter even if we fallback to binary models
            if target_rate:
                buy_threshold = max(buy_threshold, target_rate / 100.0)
                sell_threshold = max(sell_threshold, target_rate / 100.0)
            
            if model_type == 'expert':
                # Expert models might have specific tuned thresholds in their config
                # We defer to them if they are higher/specific
                buy_threshold = models.get('buy_threshold', buy_threshold)
                sell_threshold = models.get('sell_threshold', sell_threshold)
            
            signal = "WAIT"
            confidence = 0.0
            buy_prob = 0.0
            sell_prob = 0.0
            
            if model_type == 'binary' or model_type == 'expert':
                # Predict with both BUY and SELL models
                # Predict with both BUY and SELL models
                buy_prob = float(models['buy_model'].predict(X_scaled, verbose=0)[0][0])
                sell_prob = float(models['sell_model'].predict(X_scaled, verbose=0)[0][0])
                
                if buy_prob >= buy_threshold and buy_prob > sell_prob:
                    signal = "BUY"
                    confidence = buy_prob
                elif sell_prob >= sell_threshold and sell_prob > buy_prob:
                    signal = "SELL"
                    confidence = sell_prob
                    
            else:  # enhanced 3-class
                proba = models['model'].predict(X_scaled, verbose=0)[0]
                # Assuming 0=SELL, 1=WAIT, 2=BUY
                sell_prob = float(proba[0])
                wait_prob = float(proba[1])
                buy_prob = float(proba[2])
                
                if buy_prob >= self.confidence_threshold:
                    signal = "BUY"
                    confidence = buy_prob
                elif sell_prob >= self.confidence_threshold:
                    signal = "SELL"
                    confidence = sell_prob
            
            # Skip if WAIT
            if signal == "WAIT":
                # Continue to return result but don't alert/cooldown
                pass
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            
            # Calculate TP/SL
            # We need dummy TP/SL for WAIT signal or just dummy values
            levels = {
                'tp_price': 0.0, 'sl_price': 0.0,
                'tp_pips': 0, 'sl_pips': 0
            }
            if signal != "WAIT":
                # Get last ATR for dynamic levels
                atr_norm = float(features['atr_norm'].iloc[-1])
                pip_value = self.get_pip_value(symbol)
                atr_pips = (atr_norm * current_price) / pip_value
                
                # Pass ATR to calculate dynamic TP/SL
                levels = self.calculate_tp_sl(symbol, signal, current_price, atr_pips=atr_pips)
            
            # Build result
            result = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'buy_prob': buy_prob,
                'sell_prob': sell_prob,
                'price_at_signal': current_price,
                'tp_price': levels['tp_price'],
                'sl_price': levels['sl_price'],
                'tp_pips': levels['tp_pips'],
                'sl_pips': levels['sl_pips'],
                'model_version': 'binary_v1' # Keep this from original
            }
            
            if signal != "WAIT":
                logger.info(
                    f"🎯 {symbol} {signal} @ {current_price:.5f} "
                    f"(Conf: {confidence:.1%}, TP: {levels['tp_price']:.5f}, SL: {levels['sl_price']:.5f})"
                )
            else:
                logger.info(f"{symbol}: WAIT (buy={buy_prob:.2%}, sell={sell_prob:.2%})")

            if save_to_db:
                self.db.save_signal(result)
                
                # Update cooldown only if actionable signal
                if signal != "WAIT":
                    self._recent_signals[symbol] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}", exc_info=True)
            print(f"DEBUG: Exception for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
    result = engine.predict_symbol("EURUSD", save_to_db=False)
    if result:
        print(f"\n{result['signal']} {result['symbol']} @ {result['price_at_signal']:.5f}")
        print(f"TP: {result['tp_price']:.5f} (+{result['tp_pips']} pips)")
        print(f"SL: {result['sl_price']:.5f} (-{result['sl_pips']} pips)")
    else:
        print("No signal detected")
