import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger("ConfluenceMLFilter")

_INSTANCE: Optional["ConfluenceMLFilter"] = None


class ConfluenceMLFilter:
    """
    LightGBM Meta-Labeling Classifier for Confluence M15 Setups.
    Evaluates candidate trade setups at Candle 3 entry to predict the probability
    that the trade will hit 1.5R Take Profit before hitting Stop Loss.
    """

    def __init__(
        self,
        model_path: str = r"models\confluence_ml_filter.joblib",
        meta_path: str = r"models\confluence_ml_filter_meta.json",
        default_threshold: float = 0.48,
    ):
        self.model_path = model_path
        self.meta_path = meta_path
        self.threshold = default_threshold
        self.enabled = False
        self.model = None
        self.meta = {}
        self.feature_cols = []
        self.input_columns = []
        self.all_symbols = []

        self._load_model()

    def _load_model(self):
        """Safely load model artifact and schema metadata."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            logger.warning(f"⚠️ ML Filter model files not found ({self.model_path}). Filter inactive.")
            self.enabled = False
            return

        try:
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)

            self.model = joblib.load(self.model_path)
            self.input_columns = self.meta.get("input_columns", [])
            self.feature_cols = self.meta.get("feature_cols", [])
            self.all_symbols = self.meta.get("all_symbols", [])
            self.threshold = float(self.meta.get("default_threshold", self.threshold))
            self.enabled = True
            logger.info(f"🧠 Confluence ML Filter Loaded (Threshold: {self.threshold:.2f}, Features: {len(self.input_columns)}).")
        except Exception as e:
            logger.error(f"❌ Failed to load ML Filter: {e}", exc_info=True)
            self.enabled = False

    def evaluate_setup(self, setup: Dict[str, Any], mt5_inst=None) -> Dict[str, Any]:
        """
        Evaluate a candidate Confluence setup and output win probability.
        
        Returns:
            dict: {
                "evaluated": bool,
                "probability": float (0.0 to 1.0),
                "confidence_score_pct": float (0.0 to 100.0),
                "threshold": float,
                "passed": bool,
                "recommendation": "EXECUTE" | "SUPPRESS" | "BYPASS"
            }
        """
        if not self.enabled or self.model is None:
            return {
                "evaluated": False,
                "probability": 0.50,
                "confidence_score_pct": 50.0,
                "threshold": self.threshold,
                "passed": True,
                "recommendation": "BYPASS",
                "reason": "ML Filter is disabled or model not loaded."
            }

        try:
            from core.manual_model import get_mt5, get_broker_offset_hours, get_pip_size
            mt5 = mt5_inst or get_mt5()
            if not mt5:
                return {"evaluated": False, "passed": True, "probability": 0.50, "recommendation": "BYPASS"}

            symbol = setup["symbol"]
            direction = setup["direction"]
            is_buy = (direction == "BUY")
            pip_sz = get_pip_size(symbol)
            offset_h = get_broker_offset_hours(symbol)

            # Fetch recent 120 bars to compute rolling ATR, RSI, EMA
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 120)
            if rates is None or len(rates) < 85:
                return {"evaluated": False, "passed": True, "probability": 0.50, "recommendation": "BYPASS"}

            df = pd.DataFrame(rates)
            c_arr = df["close"].values
            h_arr = df["high"].values
            l_arr = df["low"].values
            o_arr = df["open"].values
            n = len(c_arr)
            i = n - 2 # Magic candle is usually completed bar (n-2) or (n-1)

            # Technical indicators
            tr = np.maximum(h_arr[1:] - l_arr[1:], np.maximum(abs(h_arr[1:] - c_arr[:-1]), abs(l_arr[1:] - c_arr[:-1])))
            atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().iloc[-1]
            atr_14_pips = atr_14 / pip_sz

            ema20 = pd.Series(c_arr).ewm(span=20, adjust=False).mean().iloc[-1]
            ema80 = pd.Series(c_arr).ewm(span=80, adjust=False).mean().iloc[-1]

            delta = pd.Series(c_arr).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-9)
            rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])

            # Setup candle geometry
            mc = setup.get("magic_candle", {})
            bc = setup.get("candle_1", setup.get("break_candle", {}))
            c2_h = float(mc.get("high", h_arr[i]))
            c2_l = float(mc.get("low", l_arr[i]))
            c2_c = float(mc.get("close", c_arr[i]))
            c2_o = float(mc.get("open", o_arr[i]))

            c1_h = float(bc.get("high", h_arr[i-1]))
            c1_l = float(bc.get("low", l_arr[i-1]))
            c1_c = float(bc.get("close", c_arr[i-1]))
            c1_o = float(bc.get("open", o_arr[i-1]))

            reclaimed_line = float(setup.get("reclaimed_line", 0.0))
            is_day_line = 1 if "Day" in str(setup.get("line_type", "")) else 0

            c2_body = abs(c2_c - c2_o)
            c2_range = max(c2_h - c2_l, 1e-6)
            c1_range = max(c1_h - c1_l, 1e-6)

            entry = float(setup.get("entry", c2_h if is_buy else c2_l))

            if is_buy:
                c1_sweep_wick = max(0, reclaimed_line - c1_l)
                c1_sweep_wick_pips = c1_sweep_wick / pip_sz
                c1_wick_ratio = c1_sweep_wick / c1_range

                c2_reclaim_wick = max(0, c2_c - c2_l)
                c2_reclaim_wick_pips = c2_reclaim_wick / pip_sz
                c2_reclaim_wick_ratio = c2_reclaim_wick / c2_range
                c2_wick_to_body_ratio = c2_reclaim_wick / max(c2_body, 1e-5)

                is_wick_only_sweep = 1.0 if c1_c >= reclaimed_line else 0.0
                is_wick_only_reclaim = 1.0 if c2_o >= reclaimed_line else 0.0

                entry_to_line_dist_pips = (entry - reclaimed_line) / pip_sz
                break_depth = max(0, reclaimed_line - min(c1_l, c2_l))
            else:
                c1_sweep_wick = max(0, c1_h - reclaimed_line)
                c1_sweep_wick_pips = c1_sweep_wick / pip_sz
                c1_wick_ratio = c1_sweep_wick / c1_range

                c2_reclaim_wick = max(0, c2_h - c2_c)
                c2_reclaim_wick_pips = c2_reclaim_wick / pip_sz
                c2_reclaim_wick_ratio = c2_reclaim_wick / c2_range
                c2_wick_to_body_ratio = c2_reclaim_wick / max(c2_body, 1e-5)

                is_wick_only_sweep = 1.0 if c1_c <= reclaimed_line else 0.0
                is_wick_only_reclaim = 1.0 if c2_o <= reclaimed_line else 0.0

                entry_to_line_dist_pips = (reclaimed_line - entry) / pip_sz
                break_depth = max(0, max(c1_h, c2_h) - reclaimed_line)

            sl_pips = float(setup.get("sl_pips", 10.0))
            sl_dist = sl_pips * pip_sz

            now_utc = pd.Timestamp.now(tz="UTC")
            hr = now_utc.hour

            feat_dict = {
                "direction": 1 if is_buy else 0,
                "is_day_line": is_day_line,
                "c2_body_pips": c2_body / pip_sz,
                "c2_range_pips": c2_range / pip_sz,
                "c2_body_ratio": c2_body / c2_range,
                "c1_range_pips": c1_range / pip_sz,
                "range_ratio_c2_c1": c2_range / c1_range,
                "break_depth_pips": break_depth / pip_sz,
                "c1_sweep_wick_pips": c1_sweep_wick_pips,
                "c1_wick_ratio": c1_wick_ratio,
                "c2_reclaim_wick_pips": c2_reclaim_wick_pips,
                "c2_reclaim_wick_ratio": c2_reclaim_wick_ratio,
                "c2_wick_to_body_ratio": c2_wick_to_body_ratio,
                "is_wick_only_sweep": is_wick_only_sweep,
                "is_wick_only_reclaim": is_wick_only_reclaim,
                "entry_to_line_dist_pips": entry_to_line_dist_pips,
                "sl_pips": sl_pips,
                "atr_14_pips": atr_14_pips,
                "sl_to_atr": sl_dist / max(atr_14, 1e-6),
                "rsi_14": rsi_14,
                "dist_ema20_pips": (c2_c - ema20) / pip_sz,
                "dist_ema80_pips": (c2_c - ema80) / pip_sz,
                "hour_utc": hr,
                "day_of_week": now_utc.weekday(),
                "is_london": 1 if 7 <= hr <= 16 else 0,
                "is_ny": 1 if 12 <= hr <= 21 else 0,
                "is_asia": 1 if (hr >= 22 or hr <= 7) else 0,
            }

            # One-hot encode symbols
            for s in self.all_symbols[1:]:
                feat_dict[f"sym_{s}"] = 1.0 if symbol == s else 0.0

            # Convert to DataFrame with exact column order
            row_df = pd.DataFrame([feat_dict])[self.input_columns]

            prob = float(self.model.predict_proba(row_df)[0, 1])
            passed = bool(prob >= self.threshold)
            score_pct = round(prob * 100, 1)

            res = {
                "evaluated": True,
                "probability": round(prob, 4),
                "confidence_score_pct": score_pct,
                "threshold": self.threshold,
                "passed": passed,
                "recommendation": "EXECUTE" if passed else "SUPPRESS",
                "sl_to_atr": round(feat_dict["sl_to_atr"], 2),
                "ema80_trend_pips": round(feat_dict["dist_ema80_pips"], 1)
            }

            if passed:
                logger.info(f"🎯 ML FILTER PASSED: {symbol} {direction} (P(Win) = {score_pct}% >= {self.threshold*100:.0f}%)")
            else:
                logger.info(f"🛡️ ML FILTER SUPPRESSED: {symbol} {direction} (P(Win) = {score_pct}% < {self.threshold*100:.0f}% threshold)")

            return res

        except Exception as e:
            logger.error(f"Error evaluating ML filter for {setup.get('symbol')}: {e}", exc_info=True)
            return {"evaluated": False, "passed": True, "probability": 0.50, "recommendation": "BYPASS"}


def get_ml_filter(default_threshold: float = 0.48) -> ConfluenceMLFilter:
    """Get singleton instance of ConfluenceMLFilter."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ConfluenceMLFilter(default_threshold=default_threshold)
    return _INSTANCE
