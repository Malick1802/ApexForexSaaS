"""
Walk-Forward Audit — Expert Models, Last 14 Days
=================================================
Uses the Phase 3 Expert adapted models (not Foundation Brain alone).
Reports accuracy per symbol, per direction, per conviction tier (60%+).
Updates the trading whitelist with results.
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# Force UTF-8 output to avoid Windows cp1252 codec crashes
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras
from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from data_pipeline.labeling import triple_barrier_label
from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WFA - %(levelname)s - %(message)s')
logger = logging.getLogger("WFA_Expert")

# ── Config ────────────────────────────────────────────────
OOS_START_DATE = "2026-04-16"   # 14-day window
TIERS          = [0.60, 0.70, 0.80, 0.90]   # 60%+ only
MIN_TRADES     = 1               # Include even 1-trade results
APPROVAL_FLOOR = 0.70            # Win rate needed for APPROVED status
APPROVAL_TRADES = 2              # Min trades for APPROVED status
FEATURE_COLS = [
    'open_norm', 'high_norm', 'low_norm', 'hl_range', 'oc_range',
    'close_ret_1', 'close_ret_5', 'close_ret_10', 'rsi', 'atr_norm',
    'bb_position', 'bb_width_norm', 'macd_norm', 'macd_signal_norm',
    'macd_hist_norm', 'volume_rel', 'volume_ret', 'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos', 'USD_strength', 'EUR_strength', 'GBP_strength',
    'JPY_strength', 'AUD_strength', 'CAD_strength', 'CHF_strength',
    'NZD_strength', 'dxy_proxy', 'dxy_ret', 'gold_ret', 'vix_proxy',
    'yield_curve_slope'
]
FEATURE_RENAMES = {
    'atr': 'atr_norm', 'bb_width': 'bb_width_norm',
    'macd': 'macd_norm', 'macd_signal': 'macd_signal_norm',
    'macd_hist': 'macd_hist_norm', 'volume_norm': 'volume_rel'
}
# ─────────────────────────────────────────────────────────


def load_foundation():
    """Load Foundation Brain + scaler as fallback."""
    brain_path  = PROJECT_ROOT / "models" / "foundation" / "foundation_brain.keras"
    scaler_path = PROJECT_ROOT / "models" / "foundation" / "scaler.joblib"
    if not brain_path.exists():
        return None, None
    model = keras.models.load_model(
        str(brain_path),
        custom_objects={
            'VariableSelectionNetwork': VariableSelectionNetwork,
            'GatedResidualNetwork':     GatedResidualNetwork
        }
    )
    scaler = joblib.load(str(scaler_path))
    return model, scaler


def load_expert(symbol, foundation_scaler):
    """Try to load the Phase-3 expert for this symbol. Falls back to Foundation."""
    expert_path = PROJECT_ROOT / "models" / "expert" / symbol / "expert_model.keras"
    if not expert_path.exists():
        return None, foundation_scaler, "foundation"
    try:
        model = keras.models.load_model(
            str(expert_path),
            custom_objects={
                'VariableSelectionNetwork': VariableSelectionNetwork,
                'GatedResidualNetwork':     GatedResidualNetwork
            }
        )
        return model, foundation_scaler, "expert"
    except Exception as e:
        logger.warning(f"  Expert load failed for {symbol}: {e}. Using Foundation.")
        return None, foundation_scaler, "foundation"


def align_features(features, df):
    """Align feature columns to match FEATURE_COLS."""
    for src, dst in FEATURE_RENAMES.items():
        if src in features.columns:
            features[dst] = features[src]

    for i in [1, 5, 10]:
        features[f'close_ret_{i}'] = df['close'].pct_change(i).fillna(0)

    ts = features.index
    features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24.0)
    features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24.0)
    features['dow_sin']  = np.sin(2 * np.pi * ts.weekday / 7.0)
    features['dow_cos']  = np.cos(2 * np.pi * ts.weekday / 7.0)
    features['volume_ret'] = features['volume_rel'].pct_change().fillna(0)

    for c in FEATURE_COLS:
        if c not in features.columns:
            features[c] = 0.0
    return features[FEATURE_COLS]


def simulate_tiers(raw_preds, y_oos, bto_oos):
    """Run walk-forward simulation for each direction and tier."""
    results = {"BUY": {}, "SELL": {}}
    for direction_idx, direction_name in [(1, 'BUY'), (2, 'SELL')]:
        for t in TIERS:
            active_until = -1
            wins = losses = pending = 0
            for i in range(len(raw_preds)):
                if i <= active_until:
                    continue
                pred_class  = int(np.argmax(raw_preds[i]))
                confidence  = float(raw_preds[i][pred_class])
                if pred_class == direction_idx and confidence >= t:
                    truth    = int(y_oos[i])
                    duration = int(bto_oos[i])
                    if i + duration >= len(y_oos):
                        pending += 1
                        active_until = len(y_oos)
                        continue
                    if truth == direction_idx:
                        wins += 1
                    else:
                        losses += 1
                    active_until = i + duration
            total = wins + losses
            accuracy = (wins / total) if total > 0 else 0.0
            results[direction_name][int(t * 100)] = {
                "accuracy": round(accuracy, 4),
                "wins":     wins,
                "losses":   losses,
                "trades":   total,
                "pending":  pending
            }
    return results


def update_whitelist(fleet_results):
    """Persist results into trading_whitelist.json."""
    whitelist_path = PROJECT_ROOT / "config" / "trading_whitelist.json"
    if not whitelist_path.exists():
        logger.warning("Whitelist not found — skipping update.")
        return 0

    with open(whitelist_path, 'r') as f:
        whitelist = json.load(f)

    updated = 0
    for symbol, tiers in fleet_results.items():
        if symbol not in whitelist["performance_matrix"]:
            whitelist["performance_matrix"][symbol] = {"BUY": {}, "SELL": {}, "ALL": {}}

        for side in ["BUY", "SELL"]:
            for tier_pct, data in tiers.get(side, {}).items():
                if data['trades'] >= MIN_TRADES:
                    approved = (data['accuracy'] >= APPROVAL_FLOOR and
                                data['trades'] >= APPROVAL_TRADES)
                    whitelist["performance_matrix"][symbol][side][str(tier_pct)] = {
                        "win_rate":     data['accuracy'],
                        "accuracy":     data['accuracy'],
                        "trades":       data['trades'],
                        "oos_trades":   data['trades'],
                        "oos_accuracy": data['accuracy'],
                        "status":       "APPROVED" if approved else "BENCHED",
                        "last_updated": datetime.now().isoformat(),
                        "source":       "Expert WFA (14d)"
                    }
                    updated += 1

    whitelist['last_updated'] = datetime.now().isoformat()
    with open(whitelist_path, 'w') as f:
        json.dump(whitelist, f, indent=2)

    return updated


def print_results(fleet_results):
    """Print a formatted results table."""
    TIERS_PCT = [int(t * 100) for t in TIERS]

    # Header
    header_tier = "  ".join(f"{'T'+str(t):>8}" for t in TIERS_PCT)
    print()
    print("=" * 100)
    print(f"  WALK-FORWARD AUDIT — Expert Models — Last 14 Days ({OOS_START_DATE} onward)")
    print(f"  Tiers: {TIERS_PCT}% conviction | Approval: {int(APPROVAL_FLOOR*100)}% win rate, {APPROVAL_TRADES}+ trades")
    print("=" * 100)
    print(f"  {'Symbol':<12} {'Dir':<5}  {header_tier}")
    print("-" * 100)

    approved_count = 0
    total_cells = 0
    summary = []

    for symbol in sorted(fleet_results.keys()):
        model_tag = fleet_results[symbol].get("_model", "?")
        for direction in ["BUY", "SELL"]:
            row = f"  {symbol:<12} {direction:<5}  "
            for t in TIERS_PCT:
                data = fleet_results[symbol][direction].get(t)
                if data and data['trades'] > 0:
                    acc    = data['accuracy']
                    trades = data['trades']
                    status = "✅" if (acc >= APPROVAL_FLOOR and trades >= APPROVAL_TRADES) else "⚠️ "
                    cell   = f"{status}{acc*100:.0f}%/{trades}t"
                    if acc >= APPROVAL_FLOOR and trades >= APPROVAL_TRADES:
                        approved_count += 1
                    total_cells += 1
                else:
                    cell = "    —    "
                row += f"{cell:>10}  "
            row += f"[{model_tag}]"
            print(row)
        print()

    print("=" * 100)
    print(f"  APPROVED cells: {approved_count} / {total_cells} ({100*approved_count/max(total_cells,1):.0f}%)")
    print(f"  ✅ = win rate ≥ {int(APPROVAL_FLOOR*100)}% with ≥ {APPROVAL_TRADES} trades")
    print("=" * 100)
    print()

    # Top performers
    top = []
    for symbol, tiers in fleet_results.items():
        for direction in ["BUY", "SELL"]:
            for t_pct in TIERS_PCT:
                data = tiers[direction].get(t_pct)
                if data and data['trades'] >= APPROVAL_TRADES and data['accuracy'] >= APPROVAL_FLOOR:
                    top.append((data['accuracy'], symbol, direction, t_pct, data['trades']))

    if top:
        top.sort(reverse=True)
        print("  🏆 TOP PERFORMERS (≥60% win rate, ≥3 trades):")
        print(f"  {'Symbol':<12} {'Dir':<5} {'Tier':>5}  {'Win Rate':>9}  {'Trades':>7}")
        print("  " + "-" * 50)
        for acc, sym, d, t, tr in top[:20]:
            print(f"  {sym:<12} {d:<5} {t:>4}%  {acc*100:>8.1f}%  {tr:>7}")
        print()


def run_wfa():
    logger.info("=" * 60)
    logger.info("  WALK-FORWARD AUDIT — Expert Models (14-day OOS)")
    logger.info("=" * 60)

    data_engine      = DataEngine()
    feature_engineer = FeatureEngineer()
    global_engineer  = GlobalFeatureEngineer()

    # Load Foundation model as base / fallback
    logger.info("Loading Foundation Brain...")
    foundation_model, foundation_scaler = load_foundation()
    if not foundation_model:
        logger.error("Foundation Brain not found — aborting.")
        return

    # Global macro context
    global_data = {}
    for g in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "GOLD", "^TNX"]:
        try:
            gdf = data_engine.fetch(g, interval="1h", days=60)
            if gdf is not None:
                global_data[g] = gdf
        except Exception:
            pass

    symbols     = data_engine.get_all_pairs()
    fleet_results = {}
    oos_cutoff  = pd.Timestamp(OOS_START_DATE, tz='UTC')

    for symbol in symbols:
        logger.info(f"Simulating {symbol}...")
        try:
            df = data_engine.fetch(symbol, interval="1h", days=60)
            if df is None or len(df) < 100:
                logger.warning(f"  Skipping {symbol} — insufficient data")
                continue

            df_labeled    = triple_barrier_label(df, symbol=symbol)
            base_features = feature_engineer.extract_features(df_labeled)
            features      = global_engineer.add_global_features(symbol, base_features, global_data)
            features      = align_features(features, df)

            y_series    = pd.Series(df_labeled['label'].astype(int).values,
                                     index=df_labeled.index)
            bto_series  = pd.Series(df_labeled['bars_to_outcome'].astype(int).values,
                                     index=df_labeled.index)

            # Align to features index (feature extraction may drop some rows)
            common_idx  = features.index.intersection(y_series.index)
            features    = features.loc[common_idx]
            y_aligned   = y_series.loc[common_idx]
            bto_aligned = bto_series.loc[common_idx]

            X, _ = feature_engineer.create_sequences(
                features, y_aligned, sequence_length=60
            )
            if len(X) == 0:
                continue

            y_all   = y_aligned.values
            bto_all = bto_aligned.values

            oos_mask = features.index[-len(X):] >= oos_cutoff
            X_oos    = X[oos_mask]
            y_oos    = y_all[-len(X):][oos_mask]
            bto_oos  = bto_all[-len(X):][oos_mask]

            if len(X_oos) == 0:
                logger.warning(f"  No OOS rows for {symbol} after {OOS_START_DATE}")
                continue

            # Scale
            X_flat   = X_oos.reshape(-1, X_oos.shape[2])
            X_scaled = foundation_scaler.transform(X_flat).reshape(len(y_oos), 60, -1)

            # Load best available model
            expert_model, _, model_tag = load_expert(symbol, foundation_scaler)
            model = expert_model if expert_model else foundation_model
            logger.info(f"  [{model_tag.upper()}] Running {len(X_oos)} OOS predictions...")

            raw_preds = model.predict(X_scaled, verbose=0)

            tier_results = simulate_tiers(raw_preds, y_oos, bto_oos)
            tier_results["_model"] = model_tag
            fleet_results[symbol]  = tier_results

        except Exception as e:
            logger.error(f"  Failed {symbol}: {e}", exc_info=True)

    # Print full table
    print_results(fleet_results)

    # Save results to JSON artifact
    results_path = PROJECT_ROOT / "logs" / "wfa_expert_14d_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        # Strip _model key before saving
        clean = {s: {k: v for k, v in d.items() if k != '_model'} for s, d in fleet_results.items()}
        json.dump({"run_date": datetime.now().isoformat(), "oos_start": OOS_START_DATE, "results": clean}, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Update whitelist
    updated = update_whitelist(fleet_results)
    logger.info(f"Whitelist updated with {updated} records.")


if __name__ == "__main__":
    run_wfa()
