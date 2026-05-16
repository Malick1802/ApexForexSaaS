"""
Foundation Brain v3 Trainer
============================
Improvements over v2:
  - Units: 32 → 64 (2x model capacity)
  - Sequence: 24h → 48h (2 full trading days of context)
  - Stride: 8 → 4 (more training samples)
  - Features: 47 → ~59 (real VIX, real DXY, Copper, BTC, session timing)
  - Real yield curve: TNX - IRX (2Y) instead of proxy
  - Session flags: London / NY / Asian hour encoding
  - Saves to models/foundation_v3/
"""

import os, gc, sys, json, logging, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("FoundationV3")

# ── Config ────────────────────────────────────────────────────
HISTORY_DAYS   = 1825      # 5 years
OOS_DAYS       = 30
VAL_DAYS       = 150
BATCH_SIZE     = 64
EPOCHS         = 60
UNITS          = 64        # 2x v2
EARLY_STOP_PAT = 8
STRIDE         = 4         # 2x more samples than v2
SEQ_LEN        = 48        # 48-hour lookback (2x v2)
DTYPE          = np.float32

FOREX_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "GBPJPY","EURJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","GBPCHF",
    "EURGBP","AUDNZD","NZDCHF","NZDCAD","CADCHF","AUDCHF","EURCAD",
    "GBPNZD","EURNZD","GBPCAD","USDSGD","EURAUD","EURCHF","GBPAUD",
    "AUDCAD", "GOLD"
]
GOLD_SYMBOL_MT5 = "GOLD"

# Extended macro universe (yfinance)
MACRO_YF = {
    "SP500":   "^GSPC",
    "OIL":     "CL=F",
    "NASDAQ":  "^IXIC",
    "TNX":     "^TNX",    # 10Y Treasury
    "IRX":     "^IRX",    # 2Y Treasury  ← NEW: real yield curve
    "VIX":     "^VIX",    # Real VIX     ← NEW
    "DXY":     "DX-Y.NYB",# Real DXY     ← UPDATED
    "COPPER":  "HG=F",    # Copper       ← NEW
    "BTC":     "BTC-USD", # Crypto risk  ← NEW
}


# ─────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────

def fetch_mt5_pair(mt5, symbol: str, days: int) -> pd.DataFrame:
    bars_needed = days * 24
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars_needed)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No MT5 data for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df = df[['open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return df[df.index >= cutoff]


def fetch_yf_macro(key: str, ticker: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(period=f"{days}d", interval="1d",
                                        auto_adjust=True, actions=False)
        if df.empty:
            logger.warning(f"yfinance: No data for {ticker}")
            return pd.DataFrame()
        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
        df = df[['Open','High','Low','Close','Volume']].rename(columns=str.lower)
        return df.resample('1h').ffill()
    except Exception as e:
        logger.warning(f"yfinance failed {ticker}: {e}")
        return pd.DataFrame()


def rolling_zscore(series: pd.Series, window: int = 720) -> pd.Series:
    mu  = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().replace(0, 1e-8)
    return (series - mu) / std


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (v3 — ~59 features)
# ─────────────────────────────────────────────────────────────

CURRENCIES = ["USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"]


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    from data_pipeline.features import FeatureEngineer
    return FeatureEngineer().extract_features(df)


def add_global_context_v3(pair_features: pd.DataFrame,
                           aligned: Dict[str, pd.DataFrame],
                           timestamp_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add 27 global context features (vs 15 in v2)."""
    f = pair_features.copy()

    # 1. Currency Strength Matrix (8)
    all_returns = {p: np.log(d['close'] / d['close'].shift(1))
                   for p, d in aligned.items() if len(d) > 1}
    ret_df = pd.DataFrame(all_returns).ffill().fillna(0)
    strength = pd.DataFrame(0.0, index=ret_df.index, columns=CURRENCIES)
    for pair in ret_df.columns:
        base, quote = pair[:3], pair[3:6]
        if base in CURRENCIES and quote in CURRENCIES:
            strength[base] += ret_df[pair]
            strength[quote] -= ret_df[pair]
    for cur in CURRENCIES:
        f[f"{cur}_strength"] = strength[cur] if cur in strength else 0.0

    # 2. DXY — prefer real DXY futures, fallback to synthetic (2)
    if "DXY" in aligned:
        dxy = aligned["DXY"]['close']
        f['dxy_level'] = rolling_zscore(dxy)
        f['dxy_ret']   = np.log(dxy / dxy.shift(1)).fillna(0)
    else:
        f['dxy_level'] = 0.0
        f['dxy_ret']   = 0.0

    # 3. Gold return (1)
    if "GOLD" in aligned:
        g = aligned["GOLD"]['close']
        f['gold_ret'] = np.log(g / g.shift(1)).fillna(0)
    else:
        f['gold_ret'] = 0.0

    # 4. Real VIX level (1) ← NEW
    if "VIX" in aligned:
        vix = aligned["VIX"]['close']
        f['vix_real'] = rolling_zscore(vix)
    else:
        f['vix_real'] = 0.0

    # 5. Yield curve slope — real TNX - IRX (1)
    if "TNX" in aligned and "IRX" in aligned:
        slope = aligned["TNX"]['close'] - aligned["IRX"]['close']
        f['yield_curve'] = rolling_zscore(slope)
    elif "TNX" in aligned:
        tnx = aligned["TNX"]['close']
        f['yield_curve'] = rolling_zscore(tnx - tnx.rolling(252).mean().fillna(method='bfill'))
    else:
        f['yield_curve'] = 0.0

    # 6. SP500 return (1)
    if "SP500" in aligned:
        sp = aligned["SP500"]['close']
        f['sp500_ret'] = np.log(sp / sp.shift(1)).fillna(0)
    else:
        f['sp500_ret'] = 0.0

    # 7. Oil return (1)
    if "OIL" in aligned:
        oil = aligned["OIL"]['close']
        f['oil_ret'] = np.log(oil / oil.shift(1)).fillna(0)
    else:
        f['oil_ret'] = 0.0

    # 8. NASDAQ return (1)
    if "NASDAQ" in aligned:
        ndx = aligned["NASDAQ"]['close']
        f['nasdaq_ret'] = np.log(ndx / ndx.shift(1)).fillna(0)
    else:
        f['nasdaq_ret'] = 0.0

    # 9. Copper return (1) ← NEW
    if "COPPER" in aligned:
        cu = aligned["COPPER"]['close']
        f['copper_ret'] = np.log(cu / cu.shift(1)).fillna(0)
    else:
        f['copper_ret'] = 0.0

    # 10. BTC return (1) ← NEW
    if "BTC" in aligned:
        btc = aligned["BTC"]['close']
        f['btc_ret'] = np.log(btc / btc.shift(1)).fillna(0)
    else:
        f['btc_ret'] = 0.0

    # 11. Session timing — cyclical encoding (4) ← NEW
    idx = f.index
    hour = idx.hour
    dow  = idx.dayofweek
    f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    f['dow_sin']  = np.sin(2 * np.pi * dow  / 5)
    f['dow_cos']  = np.cos(2 * np.pi * dow  / 5)

    # 12. Session flags (3) ← NEW: London=7-16 UTC, NY=13-21 UTC, Asia=22-6 UTC
    f['session_london'] = ((hour >= 7)  & (hour < 16)).astype(float)
    f['session_ny']     = ((hour >= 13) & (hour < 21)).astype(float)
    f['session_asia']   = ((hour >= 22) | (hour < 6)).astype(float)

    return f.ffill().fillna(0)


# ─────────────────────────────────────────────────────────────
#  LABELING
# ─────────────────────────────────────────────────────────────

def triple_barrier_label(df: pd.DataFrame, tp_pct=0.003, sl_pct=0.002, horizon=24) -> pd.Series:
    closes = df['close'].values
    labels = np.ones(len(closes), dtype=np.int32)
    for i in range(len(closes) - horizon):
        entry = closes[i]
        tp, sl = entry*(1+tp_pct), entry*(1-sl_pct)
        future = closes[i+1:i+1+horizon]
        hit_tp = np.argmax(future >= tp) if np.any(future >= tp) else -1
        hit_sl = np.argmax(future <= sl) if np.any(future <= sl) else -1
        if   hit_tp == -1 and hit_sl == -1: labels[i] = 1
        elif hit_tp == -1:                  labels[i] = 0
        elif hit_sl == -1:                  labels[i] = 2
        else: labels[i] = 2 if hit_tp <= hit_sl else 0
    return pd.Series(labels, index=df.index, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
#  DATA GENERATOR
# ─────────────────────────────────────────────────────────────

from tensorflow.keras.utils import Sequence

class ForexDataGenerator(Sequence):
    def __init__(self, features_dict, labels_dict, pairs,
                 split='train', batch_size=BATCH_SIZE, seq_len=SEQ_LEN, stride=STRIDE):
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.split      = split
        self.samples    = []
        for symbol in pairs:
            if symbol not in features_dict: continue
            n_total = len(features_dict[symbol]) - seq_len
            if n_total <= 0: continue
            n_oos = int(n_total * (OOS_DAYS  / HISTORY_DAYS))
            n_val = int(n_total * (VAL_DAYS   / HISTORY_DAYS))
            n_tr  = n_total - n_oos - n_val
            if split == 'train': start, end = 0,     n_tr
            elif split == 'val': start, end = n_tr,  n_tr + n_val
            else:                start, end = n_tr + n_val, n_total
            for i in range(start, end, stride):
                self.samples.append((symbol, i))
        if split == 'train':
            np.random.shuffle(self.samples)
        self.features_dict = features_dict
        self.labels_dict   = labels_dict

    def __len__(self): return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        n_feat = next(iter(self.features_dict.values())).shape[1]
        X = np.empty((len(batch), self.seq_len, n_feat), dtype=DTYPE)
        y = np.empty((len(batch),), dtype=np.int32)
        for i, (sym, s) in enumerate(batch):
            X[i] = self.features_dict[sym][s:s+self.seq_len]
            y[i] = self.labels_dict[sym][s+self.seq_len]
        return X, y

    def on_epoch_end(self):
        if self.split == 'train': np.random.shuffle(self.samples)


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINER
# ─────────────────────────────────────────────────────────────

class FoundationTrainerV3:
    def __init__(self):
        from core.mt5_connector import get_mt5
        self.mt5 = get_mt5()
        if self.mt5 is None:
            raise RuntimeError("MT5 not connected.")
        self.output_dir = PROJECT_ROOT / "models" / "foundation_v3"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"Fetching {len(FOREX_PAIRS)} forex pairs from MT5...")
        raw: Dict[str, pd.DataFrame] = {}
        for symbol in FOREX_PAIRS:
            try:
                raw[symbol] = fetch_mt5_pair(self.mt5, symbol, HISTORY_DAYS)
                logger.info(f"  {symbol}: {len(raw[symbol]):,} bars")
            except Exception as e:
                logger.warning(f"  {symbol}: SKIPPED — {e}")
        logger.info(f"Fetching {len(MACRO_YF)} macro assets from yfinance...")
        for key, ticker in MACRO_YF.items():
            df = fetch_yf_macro(key, ticker, HISTORY_DAYS + 60)
            if not df.empty:
                raw[key] = df
                logger.info(f"  {key} ({ticker}): {len(df):,} rows")
        return raw

    def align(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info("Aligning to common timeline...")
        common = None
        for df in raw.values():
            common = df.index if common is None else common.intersection(df.index)
        logger.info(f"  Common: {len(common):,} hours ({common[0].date()} - {common[-1].date()})")
        return {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}

    def build_corpus(self, aligned: Dict[str, pd.DataFrame]):
        logger.info("Building v3 training corpus...")
        n_features = None
        features_dict, labels_dict = {}, {}
        for symbol in FOREX_PAIRS:
            if symbol not in aligned: continue
            try:
                pair_df  = aligned[symbol]
                features = build_base_features(pair_df)
                features = add_global_context_v3(features, aligned, features.index)
                for col in features.columns:
                    features[col] = rolling_zscore(features[col])
                features = features.replace([np.inf, -np.inf], 0).fillna(0)
                if n_features is None:
                    n_features = len(features.columns)
                    logger.info(f"  Feature vector size: {n_features}")
                labels = triple_barrier_label(pair_df.reindex(features.index))
                features_dict[symbol] = features.values.astype(DTYPE)
                labels_dict[symbol]   = labels.values.astype(np.int32)
                logger.info(f"  {symbol}: {len(features_dict[symbol]):,} bars")
                del features, labels, pair_df
                gc.collect() # Force free RAM after each pair
            except Exception as e:
                logger.warning(f"  {symbol}: FAILED — {e}")
            finally:
                gc.collect()
        return features_dict, labels_dict, n_features

    def train(self, features_dict, labels_dict, n_features: int):
        import tensorflow as tf
        from tensorflow import keras
        from models.global_brain import build_global_brain

        train_gen = ForexDataGenerator(features_dict, labels_dict, FOREX_PAIRS, 'train')
        val_gen   = ForexDataGenerator(features_dict, labels_dict, FOREX_PAIRS, 'val')
        oos_gen   = ForexDataGenerator(features_dict, labels_dict, FOREX_PAIRS, 'oos')
        logger.info(f"Train: {len(train_gen.samples):,} | Val: {len(val_gen.samples):,} | OOS: {len(oos_gen.samples):,}")

        # Class weights
        subset = np.concatenate([labels_dict[s][:5000] for s in FOREX_PAIRS if s in labels_dict])
        classes, counts = np.unique(subset, return_counts=True)
        total = counts.sum()
        class_weights = {int(c): float(total/(len(classes)*cnt)) for c,cnt in zip(classes, counts)}
        logger.info(f"Class weights: {class_weights}")

        model = build_global_brain((SEQ_LEN, n_features), num_classes=3, units=UNITS)
        model.summary(print_fn=logger.info)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PAT,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                              min_lr=1e-6, verbose=1),
            keras.callbacks.ModelCheckpoint(str(self.output_dir/"foundation_brain.keras"),
                                            save_best_only=True, monitor='val_loss', verbose=1),
        ]

        logger.info("Starting v3 training...")
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                  class_weight=class_weights, callbacks=callbacks, verbose=1)

        logger.info("=" * 50)
        logger.info("OOS HOLDOUT EVALUATION")
        oos_loss, oos_acc = model.evaluate(oos_gen, verbose=0)
        logger.info(f"OOS Loss: {oos_loss:.4f} | OOS Accuracy: {oos_acc:.4f}")

        config = {
            "version": "v3",
            "oos_loss": float(oos_loss),
            "oos_accuracy": float(oos_acc),
            "n_features": n_features,
            "seq_len": SEQ_LEN,
            "units": UNITS,
            "training_pairs": FOREX_PAIRS,
            "macro_assets": list(MACRO_YF.keys()),
            "history_days": HISTORY_DAYS,
            "trained_at": datetime.now().isoformat()
        }
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved to {self.output_dir}")
        return model

    def run(self):
        logger.info("=" * 60)
        logger.info("  FOUNDATION BRAIN v3 — TRAINING START")
        logger.info(f"  Features: ~59 | Sequence: {SEQ_LEN}h | Units: {UNITS}")
        logger.info("=" * 60)
        raw     = self.fetch_all_data()
        aligned = self.align(raw)
        del raw; gc.collect()
        features_dict, labels_dict, n_features = self.build_corpus(aligned)
        del aligned; gc.collect()
        model = self.train(features_dict, labels_dict, n_features)
        del features_dict, labels_dict; gc.collect()
        logger.info("TRAINING COMPLETE — run specialist fleet pointing at foundation_v3")
        return model


if __name__ == "__main__":
    FoundationTrainerV3().run()
