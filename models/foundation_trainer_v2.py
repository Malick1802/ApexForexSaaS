"""
Foundation Brain v2 Trainer
============================
Trains the TFT on 5 years of MT5 data (forex) + yfinance (macro).
Saves to models/foundation_v2/ — leaving v1 untouched.

Key improvements over v1:
  - MT5 for forex (5-year depth, broker-accurate prices)
  - yfinance for macro (SP500, Oil, NASDAQ — daily resample to 1H)
  - Temporal train/val/OOS split (no leakage)
  - 30-day OOS holdout evaluated at the end
  - Rolling z-score scaling (no look-ahead bias)
"""

import os
import gc
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "tmp" / "v2_training"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FoundationV2")

# ── Config ────────────────────────────────────────────────────
HISTORY_DAYS   = 1825      # 5 years
OOS_DAYS       = 30        # 30-day held-out test set
VAL_DAYS       = 150       # ~5 months validation
SEQ_LEN        = 48        # 48-hour look-back window
BATCH_SIZE     = 512
EPOCHS         = 60
UNITS          = 64
EARLY_STOP_PAT = 8
STRIDE         = 4  # Sampling every 4 hours for 4x speedup

FOREX_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "GBPJPY","EURJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","GBPCHF",
    "EURGBP","AUDNZD","NZDCHF","NZDCAD","CADCHF","AUDCHF","EURCAD",
    "GBPNZD","EURNZD","GBPCAD","USDSGD","EURAUD","EURCHF","GBPAUD",
    "AUDCAD"
]
GOLD_SYMBOL_MT5 = "GOLD"   # Broker alias for XAUUSD
MACRO_YF = {               # Fetched via yfinance (daily → resampled 1H)
    "SP500":  "^GSPC",
    "OIL":    "CL=F",
    "NASDAQ": "^IXIC",
    "^TNX":   "^TNX",
}


# ─────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────

def fetch_mt5_pair(mt5, symbol: str, days: int) -> pd.DataFrame:
    """Fetch 1H bars from MT5 via copy_rates_from_pos."""
    bars_needed = days * 24  # Generous upper bound
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars_needed)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No MT5 data for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
    # Trim to requested horizon
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df.index >= cutoff]
    return df


def fetch_yf_macro(symbol_key: str, yf_ticker: str, days: int) -> pd.DataFrame:
    """Fetch daily macro data from yfinance and resample to 1H via forward-fill."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period=f"{days}d", interval="1d", auto_adjust=True, actions=False)
        if df.empty:
            logger.warning(f"yfinance: No data for {yf_ticker}")
            return pd.DataFrame()
        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
        df = df[['Open','High','Low','Close','Volume']].rename(columns=str.lower)
        # Resample to 1H by forward-filling daily candle
        df_1h = df.resample('1h').ffill()
        return df_1h
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {yf_ticker}: {e}")
        return pd.DataFrame()


def rolling_zscore(series: pd.Series, window: int = 720) -> pd.Series:
    """Rolling z-score to prevent look-ahead bias in scaling."""
    mu  = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().replace(0, 1e-8)
    return (series - mu) / std


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract OHLCV-based technical features."""
    from data_pipeline.features import FeatureEngineer
    fe = FeatureEngineer()
    features = fe.extract_features(df)
    return features


def add_global_context(pair_features: pd.DataFrame,
                       aligned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Enrich pair features with global macro context using GlobalFeatureEngineer."""
    from data_pipeline.global_features import GlobalFeatureEngineer
    ge = GlobalFeatureEngineer()
    return ge.add_global_features("", pair_features, aligned)


# ─────────────────────────────────────────────────────────────
#  LABELING
# ─────────────────────────────────────────────────────────────

def triple_barrier_label_fast(df: pd.DataFrame, tp_pct: float = 0.003,
                               sl_pct: float = 0.0015, horizon: int = 24) -> pd.Series:
    """
    Vectorised triple-barrier labeling.
    Returns: 2=BUY, 0=SELL, 1=WAIT (neutral)
    """
    closes = df['close'].values
    labels = np.ones(len(closes), dtype=np.int32)  # default WAIT

    for i in range(len(closes) - horizon):
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        future = closes[i+1 : i+1+horizon]
        hit_tp = np.argmax(future >= tp) if np.any(future >= tp) else -1
        hit_sl = np.argmax(future <= sl) if np.any(future <= sl) else -1

        if hit_tp == -1 and hit_sl == -1:
            labels[i] = 1   # WAIT
        elif hit_tp == -1:
            labels[i] = 0   # SELL
        elif hit_sl == -1:
            labels[i] = 2   # BUY
        else:
            labels[i] = 2 if hit_tp <= hit_sl else 0

    return pd.Series(labels, index=df.index, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
#  SEQUENCE BUILDER (MEMORY EFFICIENT)
# ─────────────────────────────────────────────────────────────

from tensorflow.keras.utils import Sequence

class ForexDataGenerator(Sequence):
    def __init__(self, features_dict, labels_dict, pairs, split='train', batch_size=BATCH_SIZE, seq_len=SEQ_LEN, stride=STRIDE):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.samples = [] 
        
        for symbol in pairs:
            if symbol not in features_dict: continue
            feat_len = len(features_dict[symbol])
            n_total = feat_len - seq_len
            if n_total <= 0: continue
            
            n_oos = int(n_total * (OOS_DAYS / HISTORY_DAYS))
            n_val = int(n_total * (VAL_DAYS / HISTORY_DAYS))
            n_train = n_total - n_oos - n_val
            
            if split == 'train':
                start, end = 0, n_train
            elif split == 'val':
                start, end = n_train, n_train + n_val
            else: # oos
                start, end = n_train + n_val, n_total
                
            # Stride added here to significantly reduce redundant overlapping sequences
            for i in range(start, end, stride):
                self.samples.append((symbol, i))
                
        if split == 'train':
            np.random.shuffle(self.samples)
            
        # Accept pre-converted numpy arrays directly (no copy made)
        # features_dict must be {symbol: np.float32 array}
        # labels_dict must be  {symbol: np.int32 array}
        self.features_dict = features_dict
        self.labels_dict   = labels_dict

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size : (idx + 1) * self.batch_size]
        sample_shape = next(iter(self.features_dict.values())).shape[1]
        X = np.empty((len(batch_samples), self.seq_len, sample_shape), dtype=np.float32)
        y = np.empty((len(batch_samples),), dtype=np.int32)
        
        for i, (sym, start_idx) in enumerate(batch_samples):
            X[i] = self.features_dict[sym][start_idx : start_idx + self.seq_len]
            y[i] = self.labels_dict[sym][start_idx + self.seq_len] # Label is aligned to the END of the sequence
            
        return X, y
    
    def on_epoch_end(self):
        if self.split == 'train':
            np.random.shuffle(self.samples)


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINER
# ─────────────────────────────────────────────────────────────

class FoundationTrainerV2:
    def __init__(self):
        from core.mt5_connector import get_mt5
        self.mt5 = get_mt5()
        if self.mt5 is None:
            raise RuntimeError("MT5 not connected. Open MetaTrader 5 first.")
        self.output_dir = PROJECT_ROOT / "models" / "foundation_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch all data ───────────────────────────────
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"Fetching {len(FOREX_PAIRS)+1} forex pairs from MT5 ({HISTORY_DAYS} days)...")
        raw: Dict[str, pd.DataFrame] = {}

        for symbol in FOREX_PAIRS:
            try:
                df = fetch_mt5_pair(self.mt5, symbol, HISTORY_DAYS)
                logger.info(f"  {symbol}: {len(df):,} bars | {df.index[0].date()} - {df.index[-1].date()}")
                raw[symbol] = df
            except Exception as e:
                logger.warning(f"  {symbol}: SKIPPED — {e}")

        # Gold via MT5
        try:
            df = fetch_mt5_pair(self.mt5, GOLD_SYMBOL_MT5, HISTORY_DAYS)
            raw["GOLD"] = df
            logger.info(f"  GOLD: {len(df):,} bars")
        except Exception as e:
            logger.warning(f"  GOLD: SKIPPED — {e}")

        # Macro via yfinance
        logger.info("Fetching macro context from yfinance (daily -> 1H)...")
        for key, ticker in MACRO_YF.items():
            df = fetch_yf_macro(key, ticker, HISTORY_DAYS + 60)
            if not df.empty:
                raw[key] = df
                logger.info(f"  {key} ({ticker}): {len(df):,} 1H rows")

        return raw

    # ── Step 2: Align all data to common timeline ────────────
    def align(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info("Aligning all symbols to common timeline...")
        common_index = None
        for sym, df in raw.items():
            idx = df.index
            common_index = idx if common_index is None else common_index.intersection(idx)
        logger.info(f"  Common timeline: {len(common_index):,} hours ({common_index[0].date()} - {common_index[-1].date()})")
        return {s: df.reindex(common_index).ffill().bfill() for s, df in raw.items()}

    # ── Step 3: Build training corpus ────────────────────────
    def build_corpus(self, aligned: Dict[str, pd.DataFrame]):
        logger.info("Building training corpus (rolling z-score + triple barrier labels)...")
        
        n_features = None
        features_dict = {}
        labels_dict = {}
        sample_count = 0
        
        idx = next(iter(aligned.values())).index
        total_hours = len(idx)
        oos_start_pos  = total_hours - OOS_DAYS * 24
        val_start_pos  = oos_start_pos - VAL_DAYS * 24
        
        logger.info(f"  Total timeline : {total_hours:,} hours")
        logger.info(f"  Train ends     : {idx[val_start_pos - 1].date()}")
        logger.info(f"  Val period     : {idx[val_start_pos].date()} - {idx[oos_start_pos - 1].date()}")
        logger.info(f"  OOS holdout    : {idx[oos_start_pos].date()} - {idx[-1].date()}")

        for symbol in FOREX_PAIRS:
            if symbol not in aligned:
                continue
            try:
                pair_df = aligned[symbol]
                features = build_features(pair_df, symbol)
                features = add_global_context(features, aligned)
                
                for col in features.columns:
                    features[col] = rolling_zscore(features[col])
                features = features.replace([np.inf, -np.inf], 0).fillna(0)
                
                if n_features is None:
                    n_features = len(features.columns)
                    logger.info(f"  Feature vector size: {n_features}")

                labels = triple_barrier_label_fast(pair_df.reindex(features.index))

                # ── Convert to numpy immediately and drop DataFrames ──
                # This halves peak memory vs keeping DataFrames alive
                features_dict[symbol] = features.values.astype(np.float32)
                labels_dict[symbol]   = labels.values.astype(np.int32)
                del features, labels, pair_df

                n_seq = max(0, features_dict[symbol].shape[0] - SEQ_LEN)
                sample_count += n_seq
                logger.info(f"  {symbol}: {n_seq:,} sequences")

            except Exception as e:
                logger.warning(f"  {symbol}: corpus build FAILED — {e}")
            finally:
                gc.collect()

        logger.info(f"Total sequences across all pairs: {sample_count:,}")
        return features_dict, labels_dict, n_features

    # ── Step 4: Train ────────────────────────────────────────
    def train(self, features_dict, labels_dict, n_features: int):
        import tensorflow as tf
        from tensorflow import keras
        from models.global_brain import build_global_brain

        train_gen = ForexDataGenerator(features_dict, labels_dict, FOREX_PAIRS, split='train', batch_size=BATCH_SIZE)
        val_gen   = ForexDataGenerator(features_dict, labels_dict, FOREX_PAIRS, split='val', batch_size=BATCH_SIZE)
        oos_gen   = ForexDataGenerator(features_dict, labels_dict, FOREX_PAIRS, split='oos', batch_size=BATCH_SIZE)

        logger.info(f"Split — Train: {len(train_gen.samples):,} | Val: {len(val_gen.samples):,} | OOS: {len(oos_gen.samples):,}")

        # Class weights from a subset to avoid slow full pass
        subset_y = []
        for sym in FOREX_PAIRS:
            if sym in labels_dict:
                subset_y.append(labels_dict[sym][:len(train_gen.samples)//len(FOREX_PAIRS)])
        subset_y = np.concatenate(subset_y) if subset_y else np.array([0,1,2])
        classes, counts = np.unique(subset_y, return_counts=True)
        total = counts.sum()
        class_weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        logger.info(f"Class weights: {class_weights}")

        # Build model
        input_shape = (SEQ_LEN, n_features)
        model = build_global_brain(input_shape, num_classes=3, units=UNITS)
        model.summary(print_fn=logger.info)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=EARLY_STOP_PAT, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / "foundation_brain.keras"),
                save_best_only=True, monitor='val_loss', verbose=1
            ),
        ]

        logger.info("Starting TFT model fit...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # ── OOS Evaluation ───────────────────────────────────
        logger.info("\n" + "=" * 50)
        logger.info("OOS HOLDOUT EVALUATION (30 days never seen)")
        logger.info("=" * 50)
        oos_loss, oos_acc = model.evaluate(oos_gen, verbose=0)
        logger.info(f"OOS Loss: {oos_loss:.4f} | OOS Accuracy: {oos_acc:.4f}")

        # Save OOS report
        oos_report = {
            "oos_loss": float(oos_loss),
            "oos_accuracy": float(oos_acc),
            "training_pairs": FOREX_PAIRS,
            "history_days": HISTORY_DAYS,
            "oos_days": OOS_DAYS,
            "val_days": VAL_DAYS,
            "n_features": n_features,
            "train_samples": len(train_gen.samples),
            "val_samples": len(val_gen.samples),
            "oos_samples": len(oos_gen.samples),
            "trained_at": datetime.now().isoformat()
        }
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(oos_report, f, indent=2)

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        with open(artifacts_dir / "foundation_v2_oos.md", 'w') as f:
            f.write(f"# Foundation Brain v2 — OOS Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| OOS Loss | {oos_loss:.4f} |\n")
            f.write(f"| OOS Accuracy | {oos_acc:.2%} |\n")
            f.write(f"| Training Pairs | {len(FOREX_PAIRS)} |\n")
            f.write(f"| History | {HISTORY_DAYS} days (5 years via MT5) |\n")
            f.write(f"| Features | {n_features} (incl. SP500, Oil, NASDAQ) |\n")
            f.write(f"| OOS Holdout | Last {OOS_DAYS} days |\n")

        logger.info(f"\nModel saved to: {self.output_dir}/foundation_brain.keras")
        logger.info(f"OOS report saved to: artifacts/foundation_v2_oos.md")

        return model

    def run(self):
        logger.info("=" * 60)
        logger.info("  FOUNDATION BRAIN v2 - TRAINING START")
        logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        raw     = self.fetch_all_data()
        aligned = self.align(raw)
        del raw; gc.collect()  # Free raw DataFrames before corpus build

        features_dict, labels_dict, n_features = self.build_corpus(aligned)
        del aligned; gc.collect()  # Free aligned DataFrames before training

        model   = self.train(features_dict, labels_dict, n_features)
        del features_dict, labels_dict; gc.collect()

        logger.info("=" * 60)
        logger.info("  TRAINING COMPLETE")
        logger.info("  To activate v2: set foundation.active_version: 'v2' in config.yaml")
        logger.info("  To revert:      set foundation.active_version: 'v1' in config.yaml")
        logger.info("=" * 60)
        return model


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = FoundationTrainerV2()
    trainer.run()
