"""
GMM Regime Detector — Data-Driven Market State Classifier
===========================================================
Uses a Gaussian Mixture Model (GMM) to learn regime boundaries from the
data itself — no hand-coded ADX thresholds. Trained once on historical
data, then used at inference time for fast classification.

Regimes (auto-discovered, then labeled post-hoc):
  Cluster 0: Low vol / ranging    → RANGING
  Cluster 1: Trending (directional momentum) → TRENDING
  Cluster 2: Crisis (volatility spike) → CRISIS

Features fed to GMM:
  - ADX (trend strength)
  - ATR z-score (volatility shock)
  - BB width z-score (normalised spread)
  - RSI deviation from 50 (momentum bias)
  - EMA deviation (distance from long-term mean)
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
N_COMPONENTS      = 3       # Number of GMM components (regimes)
ZSCORE_LOOKBACK   = 168     # 1 week of hourly bars
EMA_PERIOD        = 200
ADX_PERIOD        = 14
ATR_PERIOD        = 14
BB_PERIOD         = 20
RSI_PERIOD        = 14
MIN_BARS          = 220     # Minimum bars needed for reliable detection
GMM_MODEL_PATH    = Path("models/regime/gmm_model.pkl")
GMM_SCALER_PATH   = Path("models/regime/gmm_scaler.pkl")
GMM_LABELS_PATH   = Path("models/regime/gmm_labels.pkl")  # cluster→regime mapping

# --- Safety Thresholds ---
EMA_STRETCH_THRESHOLD = 1.8 # Lowered from 2.2 for higher sensitivity
RSI_CRISIS_HIGH       = 75.0
RSI_CRISIS_LOW        = 25.0

# ── Per-regime confidence thresholds ──────────────────────────────────────────
REGIME_THRESHOLDS = {
    "TRENDING":  0.65,
    "RANGING":   0.72,
    "CRISIS":    1.01,   # Blocks every trade
}


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING  = "RANGING"
    CRISIS   = "CRISIS"


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence_threshold: float
    block_trading: bool
    gmm_proba: np.ndarray        # Soft membership probabilities per cluster
    features: dict               # Raw feature values for logging
    reason: str


def _compute_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract the 5 regime features from OHLCV data."""
    if len(df) < MIN_BARS:
        return None

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # 1. ATR z-score
    if HAS_TA:
        atr_s = ta.atr(high, low, close, length=ATR_PERIOD).dropna()
    else:
        tr = pd.concat([(high - low),
                        (high - close.shift(1)).abs(),
                        (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_s = tr.rolling(ATR_PERIOD).mean().dropna()

    def zscore(s, n=ZSCORE_LOOKBACK):
        mu  = s.rolling(n).mean()
        sig = s.rolling(n).std()
        return ((s - mu) / sig.replace(0, np.nan)).fillna(0)

    atr_z = zscore(atr_s)

    # 2. BB width z-score
    if HAS_TA:
        bb = ta.bbands(close, length=BB_PERIOD, std=2)
        if bb is not None and len(bb.columns) >= 3:
            bb_w = (bb.iloc[:, 0] - bb.iloc[:, 2]).dropna()
        else:
            bb_w = (2 * 2 * close.rolling(BB_PERIOD).std()).dropna()
    else:
        bb_w = (2 * 2 * close.rolling(BB_PERIOD).std()).dropna()

    bb_z = zscore(bb_w)

    # 3. ADX
    if HAS_TA:
        adx_data = ta.adx(high, low, close, length=ADX_PERIOD)
        adx_col  = [c for c in (adx_data.columns if adx_data is not None else []) if c.startswith("ADX_")]
        adx_s    = adx_data[adx_col[0]].fillna(20) if adx_col else pd.Series(20, index=close.index)
    else:
        adx_s = pd.Series(20.0, index=close.index)

    # 4. RSI deviation from 50
    if HAS_TA:
        rsi_s = ta.rsi(close, length=RSI_PERIOD).fillna(50)
    else:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_s = 100 - 100 / (1 + rs)
        rsi_s = rsi_s.fillna(50)

    rsi_dev = (rsi_s - 50).abs()

    # 5. EMA deviation (price distance from EMA200, normalised by ATR)
    ema200 = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    ema_dev = ((close - ema200) / atr_s).fillna(0)

    # Align all series
    feats = pd.DataFrame({
        "atr_z":   atr_z,
        "bb_z":    bb_z,
        "adx":     adx_s,
        "rsi_dev": rsi_dev,
        "ema_dev": ema_dev.abs(),
    }).dropna()

    return feats


class GMMRegimeDetector:
    """
    GMM-based regime detector.
    
    Usage:
        detector = GMMRegimeDetector()
        detector.fit(historical_df)           # Train once, saves to disk
        result = detector.detect(recent_df)   # Fast inference
    """

    def __init__(self):
        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_to_regime: dict = {}   # e.g. {0: "RANGING", 1: "TRENDING", 2: "CRISIS"}
        self._load_if_exists()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_if_exists(self):
        if GMM_MODEL_PATH.exists() and GMM_SCALER_PATH.exists() and GMM_LABELS_PATH.exists():
            with open(GMM_MODEL_PATH,  "rb") as f: self.gmm    = pickle.load(f)
            with open(GMM_SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            with open(GMM_LABELS_PATH, "rb") as f: self.cluster_to_regime = pickle.load(f)
            logger.info(f"GMM Regime Detector loaded. Cluster map: {self.cluster_to_regime}")
        else:
            logger.info("GMM model not found. Call .fit(df) to train.")

    def _save(self):
        GMM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GMM_MODEL_PATH,  "wb") as f: pickle.dump(self.gmm,    f)
        with open(GMM_SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        with open(GMM_LABELS_PATH, "wb") as f: pickle.dump(self.cluster_to_regime, f)
        logger.info("GMM model saved.")

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, symbol: str = ""):
        """
        Train GMM on historical OHLCV data and auto-label clusters.

        Args:
            df:     OHLCV DataFrame (ideally 2+ years of hourly data)
            symbol: For logging only
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        logger.info(f"[GMM] Fitting on {len(df)} bars for {symbol}...")
        feats = _compute_features(df)
        if feats is None or len(feats) < 500:
            raise ValueError("Insufficient data for GMM training (need 500+ clean bars).")

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feats.values)

        self.gmm = GaussianMixture(
            n_components=N_COMPONENTS,
            covariance_type="full",
            n_init=10,
            max_iter=500,
            random_state=42
        )
        self.gmm.fit(X)

        # Auto-label clusters by their ATR z-score centroid
        # Highest ATR z-score mean → CRISIS
        # Highest ADX mean → TRENDING
        # Remainder → RANGING
        centroids = self.scaler.inverse_transform(self.gmm.means_)
        # centroids shape: (N_COMPONENTS, 5)
        # columns: [atr_z, bb_z, adx, rsi_dev, ema_dev]
        atr_z_col = 0
        adx_col   = 2

        crisis_cluster   = int(np.argmax(centroids[:, atr_z_col]))
        trending_cluster = int(np.argmax(
            [c[adx_col] if i != crisis_cluster else -1 for i, c in enumerate(centroids)]
        ))
        ranging_cluster  = [i for i in range(N_COMPONENTS)
                            if i not in (crisis_cluster, trending_cluster)][0]

        self.cluster_to_regime = {
            crisis_cluster:   "CRISIS",
            trending_cluster: "TRENDING",
            ranging_cluster:  "RANGING",
        }

        logger.info(f"[GMM] Cluster labels: {self.cluster_to_regime}")
        logger.info(f"[GMM] Centroids:\n{pd.DataFrame(centroids, columns=feats.columns)}")
        self._save()

    # ── Inference ─────────────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame, symbol: str = "") -> Optional[RegimeResult]:
        """Classify current market regime using the last row of features."""
        if self.gmm is None:
            logger.warning("[GMM] Model not trained yet. Falling back to RANGING.")
            return self._fallback()

        feats = _compute_features(df)
        if feats is None or len(feats) == 0:
            return self._fallback()

        last = feats.iloc[[-1]].values
        X    = self.scaler.transform(last)
        proba = self.gmm.predict_proba(X)[0]           # Soft membership
        cluster = int(np.argmax(proba))
        feat_dict = dict(zip(["atr_z", "bb_z", "adx", "rsi_dev", "ema_dev"],
                             feats.iloc[-1].values.tolist()))
                             
        # --- Hard Safety Override: Price Stretch & RSI Extremes ---
        ema_dev_val = abs(feat_dict.get('ema_dev', 0))
        current_rsi = feats['rsi_dev'].iloc[-1] + 50 # feats stores abs(rsi-50)
        # Actually feats stores abs(rsi-50). Let's get raw RSI from features dict if possible 
        # or just use the dev to detect extremes relative to 50.
        # rsi_dev > 25 means RSI > 75 or RSI < 25.
        rsi_dev_val = feat_dict.get('rsi_dev', 0)
        
        is_stretched = (ema_dev_val >= EMA_STRETCH_THRESHOLD)
        is_rsi_extreme = (rsi_dev_val >= 25.0) # |rsi-50| >= 25 -> 75 or 25
        
        if is_stretched or is_rsi_extreme:
            regime = MarketRegime.CRISIS
            regime_str = "CRISIS"
            reason_type = "PRICE STRETCH" if is_stretched else "RSI DANGER ZONE"
            log_val = ema_dev_val if is_stretched else (rsi_dev_val + 50)
            logger.warning(f"[GMM] {symbol}: {reason_type} OVERRIDE. Value {log_val:.1f}. Forcing CRISIS.")

        threshold  = REGIME_THRESHOLDS[regime_str]
        block      = (regime == MarketRegime.CRISIS)

        reason = (
            f"GMM cluster={cluster} ({regime_str}) | "
            f"proba={proba.tolist()} | "
            f"ADX={feat_dict['adx']:.1f} | "
            f"ATR_z={feat_dict['atr_z']:.2f}"
        )

        if block:
            logger.info(f"[GMM] {symbol}: CRISIS -- {reason}")
        else:
            logger.info(f"[GMM] {symbol}: {regime_str} | threshold={threshold:.0%} — {reason}")

        return RegimeResult(
            regime=regime,
            confidence_threshold=threshold,
            block_trading=block,
            gmm_proba=proba,
            features=feat_dict,
            reason=reason,
        )

    def is_tradeable(self, df: pd.DataFrame, symbol: str = "") -> Tuple[bool, float, Optional[RegimeResult]]:
        result = self.detect(df, symbol)
        if result is None:
            return True, 0.70, None
        return not result.block_trading, result.confidence_threshold, result

    def _fallback(self) -> RegimeResult:
        return RegimeResult(
            regime=MarketRegime.RANGING,
            confidence_threshold=0.70,
            block_trading=False,
            gmm_proba=np.array([0.0, 0.0, 1.0]),
            features={},
            reason="Fallback (model not trained)",
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
_gmm_detector: Optional[GMMRegimeDetector] = None

def get_gmm_detector() -> GMMRegimeDetector:
    global _gmm_detector
    if _gmm_detector is None:
        _gmm_detector = GMMRegimeDetector()
    return _gmm_detector
