"""
Foundation Brain Oracle
========================
Lightweight wrapper that runs the pre-trained Foundation Brain on a single
pair's data and returns per-bar BUY/SELL/WAIT probabilities.

These 3 probabilities are injected as extra features into the Specialist
training pipeline so each Specialist model learns to filter and amplify
the Foundation Brain's global market intelligence with pair-specific precision.

Architecture:
    Foundation Brain (TFT, 47-feat, 24-step)  ←  Global Intelligence
              ↓  fb_buy_prob / fb_sell_prob / fb_wait_prob
    Specialist LSTM (38-feat, 60-step)         ←  Precision Filter
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Must match Foundation Brain training config (foundation_trainer_v2.py)
FOUNDATION_PATH = Path("models/foundation_v2/foundation_brain.keras")
SEQ_LEN    = 24   # Foundation Brain lookback window
N_FEATURES = 47   # Foundation Brain feature vector size


class FoundationOracle:
    """
    Generates per-bar BUY/SELL/WAIT probability signals from the
    pre-trained Foundation Brain for use in Specialist training.
    """

    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = Path(model_path or FOUNDATION_PATH)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self._model_path.exists():
            logger.warning(f"Foundation Brain not found at {self._model_path}. Using neutral priors.")
            return None
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
        Build 47-feature input matching Foundation Brain training format.
        
        Uses rolling z-score normalization (same as foundation_trainer_v2.py).
        Global context features (CSM, DXY, macro) unavailable per-pair
        are filled with 0.0 — a neutral/unknown signal. The Foundation
        Brain will still leverage its local pattern knowledge.
        """
        from data_pipeline.features import FeatureEngineer
        fe = FeatureEngineer()

        # 32 base OHLCV features
        features = fe.extract_features(df)

        # Apply rolling z-score to match Foundation Brain training
        for col in features.columns:
            features[col] = self._rolling_zscore(features[col])

        # 15 global context features (Foundation Brain was trained with these)
        # Set to 0 = neutral when cross-pair data isn't available in worker context
        global_features = [
            'USD_strength', 'EUR_strength', 'GBP_strength', 'JPY_strength',
            'AUD_strength', 'CAD_strength', 'CHF_strength', 'NZD_strength',
            'dxy_proxy', 'dxy_ret', 'gold_ret', 'vix_proxy',
            'yield_curve_slope', 'sp500_ret', 'oil_ret', 'nasdaq_ret'
        ]

        needed = N_FEATURES - len(features.columns)
        for name in global_features[:max(0, needed)]:
            features[name] = 0.0

        features = features.replace([np.inf, -np.inf], 0).fillna(0)

        # Pad or trim to exactly N_FEATURES
        current = len(features.columns)
        if current < N_FEATURES:
            for i in range(N_FEATURES - current):
                features[f'_pad_{i}'] = 0.0
        elif current > N_FEATURES:
            features = features.iloc[:, :N_FEATURES]

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

        try:
            features = self._build_features(df)
            feat_vals  = features.values.astype(np.float32)
            feat_index = features.index

            # Build sequences of SEQ_LEN (24-step, matching Foundation Brain)
            X_list    = []
            valid_idx = []
            for i in range(SEQ_LEN, len(feat_vals)):
                X_list.append(feat_vals[i - SEQ_LEN:i])
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
