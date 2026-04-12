"""
Platt Scaling Calibration — Fixing Overconfident AI Probabilities
==================================================================
Problem: The model may say "70% sure" but historically only be right 45% of the time.
Solution: Platt Scaling fits a logistic regression on top of raw model scores,
          mapping them to real-world probability estimates.

Per-regime calibration is also supported:
  - TRENDING calibrator (separate from RANGING)
  - This ensures the 70% threshold is meaningful in all regimes

Usage:
    calibrator = PlattCalibrator()
    calibrator.fit(raw_scores, true_labels)
    calibrated_prob = calibrator.calibrate(raw_score)
    
Persistence: Saved to models/calibration/{symbol}_{direction}.pkl
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path("models/calibration")


class PlattCalibrator:
    """
    Two-stage probability calibrator:
      Stage 1: Platt Scaling (logistic regression on raw scores)
      Stage 2: Isotonic Regression (non-parametric, more flexible for large datasets)
    
    The calibrator is regime-aware — train separate calibrators per regime if desired.
    """

    def __init__(self, method: str = "platt"):
        """
        Args:
            method: "platt" (logistic) or "isotonic" (non-parametric)
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        self.method = method
        self.calibrator = None
        self.n_train_samples = 0
        self.reliability = {}   # bucket → (mean_pred, fraction_pos)

    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray):
        """
        Fit calibrator.

        Args:
            raw_scores:  1D array of model probabilities (float in [0, 1])
            true_labels: 1D array of binary labels (1 = correct prediction, 0 = wrong)
        """
        if len(raw_scores) < 30:
            logger.warning(f"Too few samples ({len(raw_scores)}) to calibrate reliably.")
            return

        raw_scores  = np.clip(raw_scores,  1e-6, 1 - 1e-6).reshape(-1, 1)
        true_labels = true_labels.astype(int)

        if self.method == "platt":
            self.calibrator = LogisticRegression(C=1e10)
            self.calibrator.fit(raw_scores, true_labels)
        else:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_scores.ravel(), true_labels)

        self.n_train_samples = len(raw_scores)

        # Reliability diagram bins
        frac_pos, mean_pred = calibration_curve(
            true_labels, raw_scores.ravel(), n_bins=10, strategy="quantile"
        )
        self.reliability = {
            "mean_pred":   mean_pred.tolist(),
            "fraction_pos": frac_pos.tolist()
        }

        # Log ECE (Expected Calibration Error)
        ece = np.mean(np.abs(frac_pos - mean_pred))
        logger.info(f"Calibrator fitted on {self.n_train_samples} samples | ECE={ece:.4f}")

    def calibrate(self, raw_score: float) -> float:
        """Map a raw model probability to a calibrated probability."""
        if self.calibrator is None:
            return raw_score   # No calibrator — pass through unchanged
        score = np.clip(raw_score, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            return float(self.calibrator.predict_proba([[score]])[0][1])
        else:
            return float(self.calibrator.predict([score])[0])

    def save(self, symbol: str, direction: str, regime: str = "ALL"):
        """Save calibrator to disk."""
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIBRATION_DIR / f"{symbol}_{direction}_{regime}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Calibrator saved: {path}")

    @staticmethod
    def load(symbol: str, direction: str, regime: str = "ALL") -> Optional["PlattCalibrator"]:
        """Load calibrator from disk, or return None."""
        path = CALIBRATION_DIR / f"{symbol}_{direction}_{regime}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            c = pickle.load(f)
        logger.info(f"Calibrator loaded: {path} | n={c.n_train_samples}")
        return c

    def reliability_report(self) -> str:
        """Human-readable calibration quality report."""
        if not self.reliability:
            return "No calibration data."
        lines = ["  Predicted → Actual (Calibration Quality)"]
        lines.append("  " + "-" * 40)
        for pred, actual in zip(self.reliability["mean_pred"],
                                self.reliability["fraction_pos"]):
            gap = actual - pred
            symbol = "✅" if abs(gap) < 0.05 else "⚠️ " if abs(gap) < 0.10 else "❌"
            lines.append(f"  {pred:.0%} predicted → {actual:.0%} actual  {symbol} (gap={gap:+.0%})")
        return "\n".join(lines)


class FleetCalibrationManager:
    """
    Manages calibrators for the entire 31-pair fleet.
    Trains calibrators from the historical signals database.
    """

    def __init__(self):
        self.calibrators: dict = {}   # (symbol, direction) → PlattCalibrator

    def train_from_database(self, db_path: str = "signals.db", min_samples: int = 30):
        """
        Train calibrators from historical closed signals in the database.
        
        Only ACTIVE/SUCCESS/FAIL outcomes are used (not WAIT).
        Labels: SUCCESS = 1, FAIL = 0
        """
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute("""
                SELECT symbol, signal, confidence, outcome
                FROM signals
                WHERE outcome IN ('SUCCESS', 'FAIL')
                AND signal IN ('BUY', 'SELL')
                AND confidence IS NOT NULL
            """).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.warning("No closed signals found in database for calibration.")
            return

        # Group by (symbol, direction)
        from collections import defaultdict
        groups: dict = defaultdict(lambda: {"scores": [], "labels": []})
        for r in rows:
            key = (r["symbol"], r["signal"])
            groups[key]["scores"].append(float(r["confidence"]))
            groups[key]["labels"].append(1 if r["outcome"] == "SUCCESS" else 0)

        trained = 0
        for (sym, direction), data in groups.items():
            n = len(data["scores"])
            if n < min_samples:
                logger.debug(f"Skipping {sym} {direction}: only {n} samples (<{min_samples})")
                continue
            cal = PlattCalibrator(method="platt")
            cal.fit(np.array(data["scores"]), np.array(data["labels"]))
            cal.save(sym, direction)
            self.calibrators[(sym, direction)] = cal
            logger.info(f"✅ Calibrated {sym} {direction}: {n} samples\n{cal.reliability_report()}")
            trained += 1

        logger.info(f"\nCalibration complete: {trained}/{len(groups)} pairs trained.")

    def calibrate(self, symbol: str, direction: str, raw_score: float) -> float:
        """Calibrate a single score. Returns raw score if no calibrator available."""
        key = (symbol, direction)
        if key not in self.calibrators:
            cal = PlattCalibrator.load(symbol, direction)
            if cal:
                self.calibrators[key] = cal
            else:
                return raw_score
        return self.calibrators[key].calibrate(raw_score)


# Module-level manager singleton
_manager: Optional[FleetCalibrationManager] = None

def get_calibration_manager() -> FleetCalibrationManager:
    global _manager
    if _manager is None:
        _manager = FleetCalibrationManager()
    return _manager
