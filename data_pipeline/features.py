# =============================================================================
# Feature Engineering Module
# =============================================================================
"""
Feature extraction and engineering for LSTM model training.

Extracts technical indicators and prepares data for sequence-based models.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import List, Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature extraction for forex LSTM models.
    
    Extracts:
    - Normalized OHLCV data
    - Technical indicators (RSI, ATR)
    - Correlated asset returns
    
    All features are designed to be stationary and normalized
    for optimal LSTM performance.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        return_periods: List[int] = None
    ):
        """
        Initialize feature engineer.
        
        Args:
            rsi_period: Period for RSI calculation
            atr_period: Period for ATR calculation
            return_periods: Periods for return calculations (default: [1, 5, 10])
        """
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.return_periods = return_periods or [1, 5, 10]
        
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
        features['close_ret_1'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in self.return_periods:
            if period > 1:
                features[f'close_ret_{period}'] = np.log(
                    df['close'] / df['close'].shift(period)
                )
        
        # ---------------------------------------------------------------------
        # Technical Indicators
        # ---------------------------------------------------------------------
        
        # RSI - already bounded [0, 100], scale to [0, 1]
        rsi = ta.rsi(df['close'], length=self.rsi_period)
        features['rsi'] = rsi / 100.0
        
        # ATR - normalize by close price
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        features['atr_norm'] = atr / df['close']
        
        # Bollinger Band position (where is price relative to bands)
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and len(bbands.columns) >= 3:
            bb_upper = bbands.iloc[:, 0]  # BBU
            bb_mid = bbands.iloc[:, 1]    # BBM
            bb_lower = bbands.iloc[:, 2]  # BBL
            bb_width = bb_upper - bb_lower
            features['bb_position'] = (df['close'] - bb_lower) / bb_width.replace(0, np.nan)
            features['bb_width_norm'] = bb_width / df['close']
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            # Normalize MACD by close price
            features['macd_norm'] = macd.iloc[:, 0] / df['close']
            features['macd_signal_norm'] = macd.iloc[:, 1] / df['close']
            features['macd_hist_norm'] = macd.iloc[:, 2] / df['close']
        
        # ---------------------------------------------------------------------
        # Volume features (if available and requested)
        # ---------------------------------------------------------------------
        if include_volume and 'volume' in df.columns and df['volume'].sum() > 0:
            # Volume relative to moving average
            vol_ma = df['volume'].rolling(20).mean()
            features['volume_rel'] = df['volume'] / vol_ma.replace(0, np.nan)
            
            # Volume change
            features['volume_ret'] = np.log(
                df['volume'].replace(0, np.nan) / 
                df['volume'].shift(1).replace(0, np.nan)
            )
        
        # ---------------------------------------------------------------------
        # Time-based features (cyclical encoding)
        # ---------------------------------------------------------------------
        if isinstance(df.index, pd.DatetimeIndex):
            # Hour of day (cyclical)
            hour = df.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Day of week (cyclical)
            dow = df.index.dayofweek
            features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        logger.info(f"Extracted {len(features.columns)} features")
        return features
    
    def add_correlated_asset(
        self,
        features: pd.DataFrame,
        corr_df: pd.DataFrame,
        asset_name: str = "corr"
    ) -> pd.DataFrame:
        """
        Add correlated asset features.
        
        Args:
            features: Existing feature DataFrame
            corr_df: OHLCV data for correlated asset
            asset_name: Prefix for feature names
            
        Returns:
            Features DataFrame with correlated asset columns added
        """
        if corr_df is None or corr_df.empty:
            logger.warning("No correlated asset data provided")
            return features
        
        # Align indices
        corr_aligned = corr_df.reindex(features.index)
        
        # Calculate returns
        features[f'{asset_name}_ret'] = np.log(
            corr_aligned['close'] / corr_aligned['close'].shift(1)
        )
        
        # Longer-term returns
        features[f'{asset_name}_ret_5'] = np.log(
            corr_aligned['close'] / corr_aligned['close'].shift(5)
        )
        
        # RSI of correlated asset
        if len(corr_aligned) > self.rsi_period:
            corr_rsi = ta.rsi(corr_aligned['close'], length=self.rsi_period)
            features[f'{asset_name}_rsi'] = corr_rsi / 100.0
        
        logger.info(f"Added {asset_name} features")
        return features
    
    def create_sequences(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        sequence_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            features: Feature DataFrame
            labels: Target labels
            sequence_length: Number of timesteps per sequence
            
        Returns:
            Tuple of (X, y) where:
            - X: shape (samples, sequence_length, num_features)
            - y: shape (samples,)
        """
        # Drop NaN rows
        combined = features.copy()
        combined['label'] = labels
        combined = combined.dropna()
        
        if len(combined) < sequence_length + 1:
            raise ValueError(
                f"Not enough data ({len(combined)} rows) for "
                f"sequence_length={sequence_length}"
            )
        
        feature_cols = [c for c in combined.columns if c != 'label']
        feature_data = combined[feature_cols].values
        label_data = combined['label'].values
        
        X, y = [], []
        
        for i in range(sequence_length, len(combined)):
            X.append(feature_data[i - sequence_length:i])
            y.append(label_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for documentation."""
        return [
            'open_norm', 'high_norm', 'low_norm',
            'hl_range', 'oc_range',
            'close_ret_1', 'close_ret_5', 'close_ret_10',
            'rsi', 'atr_norm',
            'bb_position', 'bb_width_norm',
            'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            'volume_rel', 'volume_ret',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        ]


def prepare_training_data(
    df: pd.DataFrame,
    labels: pd.Series,
    corr_df: Optional[pd.DataFrame] = None,
    sequence_length: int = 50
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to prepare training data.
    
    Args:
        df: OHLCV DataFrame
        labels: Target labels
        corr_df: Optional correlated asset data
        sequence_length: LSTM sequence length
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    engineer = FeatureEngineer()
    
    # Extract features
    features = engineer.extract_features(df)
    
    # Add correlated asset if available
    if corr_df is not None:
        features = engineer.add_correlated_asset(features, corr_df)
    
    # Create sequences
    X, y = engineer.create_sequences(features, labels, sequence_length)
    
    return X, y, list(features.columns)
