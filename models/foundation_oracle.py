"""
Foundation Brain Oracle
========================
Lightweight wrapper that runs the pre-trained Foundation Brain on a single
pair's data and returns per-bar BUY/SELL/WAIT probabilities.

These 3 probabilities are injected as extra features into the Specialist
training pipeline so each Specialist model learns to filter and amplify
the Foundation Brain's global market intelligence with pair-specific precision.

Architecture:
    Foundation Brain (TFT, N-feat, SEQ_LEN-step)  ←  Global Intelligence
              ↓  fb_buy_prob / fb_sell_prob / fb_wait_prob
    Specialist LSTM (N+3-feat, 60-step)            ←  Precision Filter

Version-aware: reads seq_len and n_features from co-located config.json
so it works with both v2 (47 feat, 24 steps) and v3 (~59 feat, 48 steps).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Default to v2; override by passing model_path pointing to a different version
FOUNDATION_PATH = Path("models/foundation_v2/foundation_brain.keras")

# Fallback constants used only when config.json is absent
_DEFAULT_SEQ_LEN    = 24
_DEFAULT_N_FEATURES = 47


class FoundationOracle:
    """
    Generates per-bar BUY/SELL/WAIT probability signals from the
    pre-trained Foundation Brain for use in Specialist training.
    """

    def __init__(self, model_path: str = None):
        self._model      = None
        self._model_path = Path(model_path or FOUNDATION_PATH)
        self._seq_len    = None   # Loaded lazily from config.json
        self._n_features = None   # Loaded lazily from config.json

    def _load_config(self):
        """Read seq_len and n_features from the model's co-located config.json."""
        config_path = self._model_path.parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                self._seq_len    = int(cfg.get('seq_len',    _DEFAULT_SEQ_LEN))
                self._n_features = int(cfg.get('n_features', _DEFAULT_N_FEATURES))
                logger.info(f"[Foundation Oracle] Config: seq_len={self._seq_len}, "
                            f"n_features={self._n_features}")
                return
            except Exception as e:
                logger.warning(f"[Foundation Oracle] Could not read config.json: {e}")
        # Fallback to defaults
        self._seq_len    = _DEFAULT_SEQ_LEN
        self._n_features = _DEFAULT_N_FEATURES
        logger.warning(f"[Foundation Oracle] Using defaults: seq_len={self._seq_len}, "
                       f"n_features={self._n_features}")

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self._model_path.exists():
            logger.warning(f"Foundation Brain not found at {self._model_path}. Using neutral priors.")
            return None
        self._load_config()   # Always read config before loading model
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self._model_path))
            logger.info(f"[Foundation Oracle] Brain loaded from {self._model_path}")
        except Exception as e:
            logger.warning(f"[Foundation Oracle] Failed to load: {e}")
            self._model = None
        return self._model

    def _rolling_zscore(self, series: pd.Series, window: int = 720) -> pd.Series:
        """Rolling z-score normalization — matches Foundation Brain training preprocessing."""
        mu  = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0, 1e-8)
        return (series - mu) / std

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build N_FEATURES-feature input matching Foundation Brain training format.

        Uses rolling z-score normalization. Global context features that
        require cross-pair data are filled with 0.0 (neutral) when not
        available in per-pair worker context.

        Works with any Foundation Brain version — target feature count is
        read from the model's config.json via self._n_features.
        """
        from data_pipeline.features import FeatureEngineer
        n_target = self._n_features or _DEFAULT_N_FEATURES

        fe = FeatureEngineer()
        features = fe.extract_features(df)

        # Apply rolling z-score to match Foundation Brain training
        for col in features.columns:
            features[col] = self._rolling_zscore(features[col])

        # Global context placeholders (neutral = 0.0)
        global_feature_names = [
            'USD_strength', 'EUR_strength', 'GBP_strength', 'JPY_strength',
            'AUD_strength', 'CAD_strength', 'CHF_strength', 'NZD_strength',
            'dxy_level', 'dxy_ret', 'gold_ret', 'vix_real',
            'yield_curve', 'sp500_ret', 'oil_ret', 'nasdaq_ret',
            'copper_ret', 'btc_ret',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'session_london', 'session_ny', 'session_asia',
        ]

        needed = n_target - len(features.columns)
        for name in global_feature_names[:max(0, needed)]:
            features[name] = 0.0

        features = features.replace([np.inf, -np.inf], 0).fillna(0)

        # Pad or trim to exactly n_target
        current = len(features.columns)
        if current < n_target:
            for i in range(n_target - current):
                features[f'_pad_{i}'] = 0.0
        elif current > n_target:
            features = features.iloc[:, :n_target]

        return features

    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run Foundation Brain on each bar and return BUY/SELL/WAIT probabilities.

        Args:
            df: OHLCV DataFrame (must have 'open','high','low','close','volume')

        Returns:
            DataFrame with columns [fb_buy_prob, fb_sell_prob, fb_wait_prob]
            aligned to df.index. Falls back to neutral priors (0.333) on failure.
        """
        neutral = pd.DataFrame({
            'fb_buy_prob':  0.333,
            'fb_sell_prob': 0.333,
            'fb_wait_prob': 0.334,
        }, index=df.index)

        model = self._load_model()
        if model is None:
            return neutral

        seq_len = self._seq_len or _DEFAULT_SEQ_LEN

        try:
            features = self._build_features(df)
            feat_vals  = features.values.astype(np.float32)
            feat_index = features.index

            # Build sequences using version-correct seq_len
            X_list    = []
            valid_idx = []
            for i in range(seq_len, len(feat_vals)):
                X_list.append(feat_vals[i - seq_len:i])
                valid_idx.append(feat_index[i])

            if not X_list:
                logger.warning("[Foundation Oracle] Not enough data for sequences.")
                return neutral

            X = np.array(X_list, dtype=np.float32)

            # Run Foundation Brain (3-class softmax output)
            probs = model.predict(X, verbose=0, batch_size=512)

            # Label mapping from triple_barrier_label_fast in foundation_trainer_v2.py:
            #   2 = BUY,  0 = SELL,  1 = WAIT
            # Keras one-hot encoding maps class index → column index:
            #   column 0 = SELL,  column 1 = WAIT,  column 2 = BUY
            result = pd.DataFrame({
                'fb_sell_prob': probs[:, 0],
                'fb_wait_prob': probs[:, 1],
                'fb_buy_prob':  probs[:, 2],
            }, index=valid_idx)

            # Align to original df index; back-fill the first SEQ_LEN rows
            result = result.reindex(df.index).bfill().fillna(0.333)

            buy_mean  = result['fb_buy_prob'].mean()
            sell_mean = result['fb_sell_prob'].mean()
            logger.info(
                f"[Foundation Oracle] {len(result)} predictions | "
                f"Avg BUY={buy_mean:.3f}  SELL={sell_mean:.3f}"
            )
            return result

        except Exception as e:
            logger.warning(f"[Foundation Oracle] Prediction failed: {e}. Using neutral priors.")
            return neutral
