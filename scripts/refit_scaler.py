"""
Scaler Re-Fit Script
=====================
The existing scaler.joblib was trained on a different (older) feature pipeline.
This script:
  1. Fetches 90 days of 1H data for all G7+ pairs via MT5
  2. Runs through the EXACT same 34-feature alignment used by inference.py
  3. Re-fits a StandardScaler on all collected features
  4. Saves the new scaler, backing up the old one
  5. Runs a quick bias sanity-check to confirm Z-scores are in [-3, 3] range

Run: python scripts/refit_scaler.py
"""

import sys
import os
import logging
import shutil
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("ScalerRefit")

# ── Exact 34-feature column order ────────────────────────────────────────────
F_COLS = [
    'open_norm', 'high_norm', 'low_norm', 'hl_range', 'oc_range',
    'close_ret_1', 'close_ret_5', 'close_ret_10', 'rsi', 'atr_norm',
    'bb_position', 'bb_width_norm', 'macd_norm', 'macd_signal_norm',
    'macd_hist_norm', 'volume_rel', 'volume_ret', 'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos', 'USD_strength', 'EUR_strength', 'GBP_strength',
    'JPY_strength', 'AUD_strength', 'CAD_strength', 'CHF_strength',
    'NZD_strength', 'dxy_proxy', 'dxy_ret', 'gold_ret', 'vix_proxy',
    'yield_curve_slope'
]

RENAME_MAP = {
    'atr': 'atr_norm',
    'bb_width': 'bb_width_norm',
    'macd': 'macd_norm',
    'macd_signal': 'macd_signal_norm',
    'macd_hist': 'macd_hist_norm',
    'volume_norm': 'volume_rel',
}

def align_features(df_raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same feature alignment used by inference.py."""
    # Apply rename map
    for src, dst in RENAME_MAP.items():
        if src in features.columns:
            features[dst] = features[src]

    # Explicit close returns (pct_change, not log)
    for i in [1, 5, 10]:
        features[f'close_ret_{i}'] = df_raw['close'].pct_change(i).fillna(0)

    # Cyclical time features
    ts = features.index
    features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24.0)
    features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24.0)
    features['dow_sin']  = np.sin(2 * np.pi * ts.weekday / 7.0)
    features['dow_cos']  = np.cos(2 * np.pi * ts.weekday / 7.0)

    # volume_rel fallback
    if 'volume_norm' in features.columns:
        features['volume_rel'] = features['volume_norm']
    if 'volume_ret' not in features.columns:
        if 'volume_rel' in features.columns:
            features['volume_ret'] = features['volume_rel'].pct_change().fillna(0)
        else:
            features['volume_ret'] = 0.0

    # Guarantee all 34 cols exist
    for c in F_COLS:
        if c not in features.columns:
            features[c] = 0.0

    return features[F_COLS]


def main():
    logger.info("=" * 60)
    logger.info("  Foundation Brain — Scaler Re-Fit")
    logger.info("=" * 60)

    data_engine    = DataEngine()
    feat_eng       = FeatureEngineer()
    global_eng     = GlobalFeatureEngineer()

    scaler_path    = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
    scaler_backup  = PROJECT_ROOT / "models" / "foundation" / "scaler_backup.joblib"

    # ── 1. Backup old scaler ────────────────────────────────────────────────
    if scaler_path.exists():
        shutil.copy(scaler_path, scaler_backup)
        logger.info(f"Old scaler backed up → {scaler_backup.name}")

    # ── 2. Fetch global context data (90 days, 1H) ─────────────────────────
    g7_pairs = [
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD", "GOLD", "^TNX", "^IRX"
    ]
    logger.info("Fetching global context pairs (90d)...")
    global_data = {}
    for sym in g7_pairs:
        try:
            df = data_engine.fetch(sym, interval="1h", days=90, use_cache=False)
            if df is not None and len(df) > 50:
                global_data[sym] = df
                logger.info(f"  ✓ {sym}: {len(df)} bars")
            else:
                logger.warning(f"  ✗ {sym}: insufficient data")
        except Exception as e:
            logger.warning(f"  ✗ {sym}: {e}")

    # ── 3. Collect features for all tradeable pairs ─────────────────────────
    all_symbols = data_engine.get_all_pairs()
    all_feature_blocks = []

    logger.info(f"\nProcessing {len(all_symbols)} symbols...")
    for symbol in all_symbols:
        try:
            df_raw = data_engine.fetch(symbol, interval="1h", days=90, use_cache=False)
            if df_raw is None or len(df_raw) < 100:
                logger.warning(f"  Skipping {symbol}: insufficient data ({len(df_raw) if df_raw is not None else 0} bars)")
                continue

            base_feats = feat_eng.extract_features(df_raw)
            enriched   = global_eng.add_global_features(symbol, base_feats, global_data)
            aligned    = align_features(df_raw, enriched)

            # Drop any rows with inf / NaN after alignment
            aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()

            all_feature_blocks.append(aligned)
            logger.info(f"  ✓ {symbol}: {len(aligned)} feature rows")

        except Exception as e:
            logger.warning(f"  ✗ {symbol}: {e}")

    if not all_feature_blocks:
        logger.error("No features collected — aborting.")
        return

    # ── 4. Concatenate & fit scaler ─────────────────────────────────────────
    all_features = pd.concat(all_feature_blocks, axis=0)
    logger.info(f"\nTotal rows for scaler fit: {len(all_features):,}")
    logger.info(f"Feature columns: {len(all_features.columns)} (expected 34)")

    # Pre-fit sanity check
    logger.info("\nRaw feature statistics (before scaling):")
    stats = all_features.describe().loc[['mean', 'std', 'min', 'max']]
    for col in F_COLS[:10]:  # Print first 10
        logger.info(f"  {col:20s}  mean={stats[col]['mean']:.4f}  std={stats[col]['std']:.4f}  range=[{stats[col]['min']:.4f}, {stats[col]['max']:.4f}]")

    scaler = StandardScaler()
    scaler.fit(all_features.values)

    # ── 5. Post-fit sanity check ─────────────────────────────────────────────
    sample = all_features.values[:1000]
    scaled = scaler.transform(sample)
    z_means = np.abs(scaled.mean(axis=0))
    z_stds  = scaled.std(axis=0)

    logger.info("\nPost-fit Z-score check (sample of 1000 rows):")
    for i, col in enumerate(F_COLS):
        status = "✓" if z_means[i] < 0.5 and 0.5 < z_stds[i] < 2.0 else "⚠"
        logger.info(f"  {status} {col:25s}  |z_mean|={z_means[i]:.3f}  z_std={z_stds[i]:.3f}")

    # ── 6. Save new scaler ───────────────────────────────────────────────────
    joblib.dump(scaler, scaler_path)
    logger.info(f"\n✅ New scaler saved → {scaler_path}")
    logger.info(f"   Fit on {len(all_features):,} rows × {len(F_COLS)} features")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")

    # ── 7. Quick inference bias check ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Running post-fix bias check...")
    logger.info("=" * 60)

    try:
        from tensorflow import keras
        from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork

        brain_path = PROJECT_ROOT / "models" / "foundation" / "foundation_brain.keras"
        model = keras.models.load_model(
            str(brain_path),
            custom_objects={
                'VariableSelectionNetwork': VariableSelectionNetwork,
                'GatedResidualNetwork': GatedResidualNetwork
            }
        )
        new_scaler = joblib.load(scaler_path)
        new_mean   = new_scaler.mean_
        new_scale  = new_scaler.scale_

        test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "GOLD"]
        for sym in test_symbols:
            if sym not in [s for block in all_feature_blocks for s in [sym]]:
                continue
            try:
                df_raw = global_data.get(sym)
                if df_raw is None:
                    df_raw = data_engine.fetch(sym, interval="1h", days=30)
                base_feats = feat_eng.extract_features(df_raw)
                enriched   = global_eng.add_global_features(sym, base_feats, global_data)
                aligned    = align_features(df_raw, enriched)
                aligned    = aligned.replace([np.inf, -np.inf], np.nan).dropna()

                X, _ = feat_eng.create_sequences(
                    aligned,
                    pd.Series(0, index=aligned.index),
                    sequence_length=60
                )
                if len(X) == 0:
                    continue

                X_last   = X[-1:].reshape(1, 60, 34)
                X_scaled = (X_last - new_mean) / new_scale

                # Check Z-scores
                z = X_scaled[0, -1, :]
                max_z = np.max(np.abs(z))

                preds = model.predict(X_scaled, verbose=0)[0]
                logger.info(
                    f"  {sym:8s}  WAIT={preds[0]:.3f} BUY={preds[1]:.3f} SELL={preds[2]:.3f}  "
                    f"max|Z|={max_z:.2f}"
                )
            except Exception as e:
                logger.warning(f"  {sym}: bias check failed — {e}")

    except Exception as e:
        logger.warning(f"Bias check skipped: {e}")

    logger.info("\n✅ Scaler re-fit complete. Restart the sentinel to apply changes.")


if __name__ == "__main__":
    main()
