# =============================================================================
# Feature Engineering Module
# =============================================================================
"""
Feature extraction and engineering for LSTM model training.

Extracts technical indicators and prepares data for sequence-based models.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

try:
    import pandas_ta as ta
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    try:
        import pandas_ta_classic as ta
        _PANDAS_TA_AVAILABLE = True
    except ImportError:
        ta = None
        _PANDAS_TA_AVAILABLE = False



logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature extraction for forex LSTM models.
    
    Normalizes and engineers features from OHLCV data,
    including technical indicators and custom derived features.
    """
    
    def __init__(self):
        self.feature_names = []

    def extract_features(
        self,
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from OHLCV data.
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            include_volume: Whether to include volume features
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=df.index)
        
        # Validate input
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure pandas_ta is installed before attempting to use ta methods
        if not _PANDAS_TA_AVAILABLE:
            raise ImportError(
                "pandas_ta is required for feature generation. Please install it with:\n"
                "venv\\Scripts\\pip install pandas-ta"
            )

        # ---------------------------------------------------------------------
        # Price-based features (normalized by close)
        # ---------------------------------------------------------------------
        features['open_norm'] = df['open'] / df['close']
        features['high_norm'] = df['high'] / df['close']
        features['low_norm'] = df['low'] / df['close']
        
        # Price range features
        features['hl_range'] = (df['high'] - df['low']) / df['close']
        features['oc_range'] = (df['close'] - df['open']) / df['close']
        
        # ---------------------------------------------------------------------
        # Returns (log returns for stationarity)
        # ---------------------------------------------------------------------
        features['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        features['log_return_3'] = np.log(df['close'] / df['close'].shift(3))
        features['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        features['log_return_10'] = np.log(df['close'] / df['close'].shift(10))

        # ---------------------------------------------------------------------
        # Trend Indicators
        # ---------------------------------------------------------------------
        features['sma_5'] = df.ta.sma(length=5) / df['close']
        features['sma_20'] = df.ta.sma(length=20) / df['close']
        features['ema_8'] = df.ta.ema(length=8) / df['close']
        features['ema_21'] = df.ta.ema(length=21) / df['close']
        features['ema_cross'] = features['ema_8'] - features['ema_21']
        
        # ---------------------------------------------------------------------
        # Momentum Indicators
        # ---------------------------------------------------------------------
        rsi = df.ta.rsi(length=14)
        features['rsi'] = rsi / 100.0  # Normalize to [0, 1]
        
        stoch = df.ta.stoch(k=14, d=3)
        if stoch is not None and not stoch.empty:
            features['stoch_k'] = stoch.iloc[:, 0] / 100.0
            features['stoch_d'] = stoch.iloc[:, 1] / 100.0
        
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            features['macd'] = macd.iloc[:, 0] / df['close']
            features['macd_signal'] = macd.iloc[:, 1] / df['close']
            features['macd_hist'] = macd.iloc[:, 2] / df['close']
        
        # CCI Indicator
        cci = df.ta.cci(length=14)
        if cci is not None:
            features['cci'] = cci / 200.0  # Normalize
        
        # ---------------------------------------------------------------------
        # Volatility Indicators
        # ---------------------------------------------------------------------
        bb = df.ta.bbands(length=20, std=2)
        if bb is not None and not bb.empty:
            bb_upper = bb.iloc[:, 0]
            bb_mid = bb.iloc[:, 1]
            bb_lower = bb.iloc[:, 2]
            bb_width = (bb_upper - bb_lower) / bb_mid
            bb_pos = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features['bb_width'] = bb_width
            features['bb_position'] = bb_pos
        
        atr = df.ta.atr(length=14)
        if atr is not None:
            features['atr'] = atr / df['close']
        
        # VIX proxy: Realized vol over 5-bar window
        features['vix_proxy'] = features['log_return_1'].rolling(5).std() * np.sqrt(252)
        
        # ---------------------------------------------------------------------
        # Volume Features (if available)
        # ---------------------------------------------------------------------
        if include_volume and 'volume' in df.columns:
            vol = df['volume'].replace(0, np.nan)
            features['volume_norm'] = vol / vol.rolling(20).mean()
            features['volume_log'] = np.log1p(vol)
            obv = df.ta.obv()
            if obv is not None:
                features['obv_norm'] = obv / obv.rolling(20).mean()
        
        # ---------------------------------------------------------------------
        # Market Structure
        # ---------------------------------------------------------------------
        # Swing high/low proximity
        features['high_20'] = df['high'].rolling(20).max() / df['close']
        features['low_20'] = df['low'].rolling(20).min() / df['close']
        
        # Momentum
        features['momentum_10'] = df['close'].pct_change(10)
        features['momentum_20'] = df['close'].pct_change(20)
        
        # ---------------------------------------------------------------------
        # Drop NaN rows
        # ---------------------------------------------------------------------
        features = features.dropna()
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Created sequences: X=(PENDING, PENDING, {len(self.feature_names)}), y=(PENDING,)")
        
        return features
    
    def create_sequences(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        sequence_length: int = 60
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer models.
        
        Args:
            features: Feature DataFrame (rows = timesteps)
            labels: Target labels Series
            sequence_length: Number of historical bars to include
            
        Returns:
            Tuple of (X, y) arrays
        """
        X_list, y_list = [], []
        
        feature_vals = features.values
        label_vals = labels.values
        
        # Align by index
        common_idx = features.index.intersection(labels.index)
        feature_vals = features.loc[common_idx].values
        label_vals = labels.loc[common_idx].values
        
        for i in range(sequence_length, len(feature_vals)):
            X_list.append(feature_vals[i - sequence_length:i])
            y_list.append(label_vals[i])
        
        if not X_list:
            return np.array([]), np.array([])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names from last extraction."""
        return self.feature_names
    
    def get_feature_count(self) -> int:
        """Return the number of features."""
        return len(self.feature_names)
