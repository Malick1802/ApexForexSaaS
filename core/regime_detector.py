"""
Regime Detector — Institutional Market State Classifier
=========================================================
Classifies the current market into one of four regimes BEFORE
the inference engine emits a signal, and adjusts the confidence
threshold accordingly.

Regimes:
  TRENDING_UP   — Strong uptrend (ADX > 25, price > EMA200)
  TRENDING_DOWN — Strong downtrend (ADX > 25, price < EMA200)
  RANGING       — Low directional conviction (ADX < 20)
  CRISIS        — Abnormal volatility spike (ATR z-score or BB z-score > 2.5)

Threshold Rules:
  CRISIS        → BLOCK all trades (return None)
  TRENDING      → Lower bar: 0.65 (model is more reliable in directional moves)
  RANGING       → Higher bar: 0.72 (model is less reliable in chop)

Walk-Forward Rationale:
  Window 3 (Banking Crisis) produced 27.6% OOS accuracy.
  With CRISIS detection active, those trades would have been blocked,
  raising effective OOS accuracy to approximately 40-48% on remaining windows.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional

try:
    import pandas_ta as ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    CRISIS        = "CRISIS"


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence_threshold: float   # Dynamic threshold to use for this regime
    block_trading: bool           # True = do NOT trade in this regime
    adx: float
    atr_zscore: float
    bb_zscore: float
    ema_trend: str                # "above" | "below" | "unknown"
    reason: str                   # Human-readable explanation


# ── Per-regime threshold map ──────────────────────────────────────────────────
REGIME_THRESHOLDS = {
    MarketRegime.TRENDING_UP:   0.65,
    MarketRegime.TRENDING_DOWN: 0.65,
    MarketRegime.RANGING:       0.72,
    MarketRegime.CRISIS:        1.01,   # Effectively blocks every trade
}

# ── Regime detection parameters ───────────────────────────────────────────────
ADX_TREND_THRESHOLD   = 25.0
ADX_RANGING_THRESHOLD = 20.0
ATR_ZSCORE_CRISIS     = 3.5     # Increased from 2.0 to avoid false positive "Crisis" on normal volatility
BB_ZSCORE_CRISIS      = 3.0     # Increased from 1.8 for institutional stability
EMA_PERIOD            = 200
ADX_PERIOD            = 14
ATR_PERIOD            = 14
BB_PERIOD             = 20
ZSCORE_LOOKBACK       = 100     # Increased from 24 for a more stable baseline
EMA_STRETCH_THRESHOLD = 2.5     # Price deviation in ATR units
RSI_CRISIS_HIGH       = 80.0
RSI_CRISIS_LOW        = 20.0


class RegimeDetector:
    """
    Stateless regime classifier.  Call detect(df) with an OHLCV DataFrame
    to get a RegimeResult describing the current market state.
    """

    def __init__(self):
        pass

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame, symbol: str = "") -> Optional[RegimeResult]:
        """
        Classify current market regime from an OHLCV DataFrame.

        Args:
            df:     OHLCV DataFrame (hourly, at least 250 bars recommended)
            symbol: Used only for logging

        Returns:
            RegimeResult or None if insufficient data
        """
        if df is None or len(df) < max(EMA_PERIOD + 10, ZSCORE_LOOKBACK + 10):
            logger.debug(f"[Regime] {symbol}: insufficient bars ({len(df) if df is not None else 0})")
            return None

        try:
            return self._classify(df, symbol)
        except Exception as e:
            logger.warning(f"[Regime] {symbol}: detection failed — {e}")
            return None

    def is_tradeable(self, df: pd.DataFrame, symbol: str = "") -> tuple:
        """
        Convenience wrapper.

        Returns:
            (is_tradeable: bool, threshold: float, regime: RegimeResult | None)
        """
        result = self.detect(df, symbol)
        if result is None:
            # Can't determine regime — use conservative default
            return True, 0.70, None
        return not result.block_trading, result.confidence_threshold, result

    # ── Internal logic ────────────────────────────────────────────────────────
    def _classify(self, df: pd.DataFrame, symbol: str) -> RegimeResult:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── 1. ATR z-score (volatility shock detector) ────────────────────────
        if ta:
            atr_series = ta.atr(high, low, close, length=ATR_PERIOD)
        else:
            # Manual ATR fallback (simplified)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(ATR_PERIOD).mean()

        atr_series = atr_series.dropna()
        if len(atr_series) < ZSCORE_LOOKBACK:
            atr_zscore = 0.0
        else:
            window = atr_series.iloc[-ZSCORE_LOOKBACK:]
            mu, sigma = window.mean(), window.std()
            atr_zscore = float((atr_series.iloc[-1] - mu) / sigma) if sigma > 0 else 0.0

        # ── 2. Bollinger Band width z-score ───────────────────────────────────
        if ta:
            bb = ta.bbands(close, length=BB_PERIOD, std=2)
            if bb is not None and len(bb.columns) >= 3:
                bbu = bb.iloc[:, 0]
                bbl = bb.iloc[:, 2]
                bb_width = (bbu - bbl).dropna()
            else:
                bb_width = pd.Series(dtype=float)
        else:
            sma   = close.rolling(BB_PERIOD).mean()
            std   = close.rolling(BB_PERIOD).std()
            bb_width = (2 * 2 * std).dropna()

        if len(bb_width) < ZSCORE_LOOKBACK:
            bb_zscore = 0.0
        else:
            window = bb_width.iloc[-ZSCORE_LOOKBACK:]
            mu, sigma = window.mean(), window.std()
            bb_zscore = float((bb_width.iloc[-1] - mu) / sigma) if sigma > 0 else 0.0

        # ── 3. ADX (trend strength) ────────────────────────────────────────────
        if ta:
            adx_data = ta.adx(high, low, close, length=ADX_PERIOD)
            if adx_data is not None and not adx_data.empty:
                adx_col = [c for c in adx_data.columns if c.startswith("ADX_")]
                adx = float(adx_data[adx_col[0]].iloc[-1]) if adx_col else 0.0
            else:
                adx = 0.0
        else:
            # Approximate ADX via TR smoothing (rough)
            adx = 20.0  # Neutral fallback

        # ── 4. EMA-200 trend direction ─────────────────────────────────────────
        ema200 = close.ewm(span=EMA_PERIOD, adjust=False).mean()
        current_price = float(close.iloc[-1])
        ema_val = float(ema200.iloc[-1])
        ema_trend = "above" if current_price > ema_val else "below"
        
        # ── 5. Price Stretch (Deviation from mean) ─────────────────────────────
        # If price is more than 3 ATRs away from its 200-hour EMA, it is overextended.
        price_dist = abs(current_price - ema_val)
        current_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
        # Normalise distance in 'ATR units'
        atr_distance = price_dist / current_atr if current_atr > 0 else 0.0
        
        # ── 6. RSI (Momentum Extreme) ─────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        current_rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50.0

        # ── 7. Regime classification rules ────────────────────────────────────
        # Rule 1 — CRISIS: ATR spike OR Extreme Price Stretch
        is_shock = (atr_zscore >= ATR_ZSCORE_CRISIS or bb_zscore >= BB_ZSCORE_CRISIS)
        is_stretched = (atr_distance >= EMA_STRETCH_THRESHOLD)
        is_rsi_extreme = (current_rsi >= RSI_CRISIS_HIGH or current_rsi <= RSI_CRISIS_LOW)
        
        if is_shock or is_stretched or is_rsi_extreme:
            regime = MarketRegime.CRISIS
            if is_shock:
                reason = (
                    f"VOLATILITY SHOCK: ATR z-score={atr_zscore:.2f} "
                    f"(threshold {ATR_ZSCORE_CRISIS}) | "
                    f"BB z-score={bb_zscore:.2f}"
                )
            elif is_rsi_extreme:
                reason = f"RSI DANGER ZONE: RSI is {current_rsi:.1f} (Overextended @ {RSI_CRISIS_HIGH}/{RSI_CRISIS_LOW} boundary)"
            else:
                reason = (
                    f"PRICE STRETCH: Distance from EMA200 is {atr_distance:.1f} ATRs "
                    f"(Threshold {EMA_STRETCH_THRESHOLD}). Market overextended."
                )
            logger.info(f"[Regime] {symbol}: 🚨 CRISIS detected — {reason}")

        # Rule 2 — TRENDING: ADX strong AND price clearly away from EMA200
        elif adx >= ADX_TREND_THRESHOLD:
            if ema_trend == "above":
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            reason = f"ADX={adx:.1f} > {ADX_TREND_THRESHOLD} | price {ema_trend} EMA200"
            logger.info(f"[Regime] {symbol}: 📈 {regime.value} — {reason}")

        # Rule 3 — RANGING: low ADX
        else:
            regime = MarketRegime.RANGING
            reason = f"ADX={adx:.1f} < {ADX_TREND_THRESHOLD} | choppy/ranging"
            logger.info(f"[Regime] {symbol}: ↔️  RANGING — {reason}")

        threshold = REGIME_THRESHOLDS[regime]
        block     = (regime == MarketRegime.CRISIS)

        return RegimeResult(
            regime=regime,
            confidence_threshold=threshold,
            block_trading=block,
            adx=adx,
            atr_zscore=atr_zscore,
            bb_zscore=bb_zscore,
            ema_trend=ema_trend,
            reason=reason,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
_detector: Optional[RegimeDetector] = None

def get_detector() -> RegimeDetector:
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector
