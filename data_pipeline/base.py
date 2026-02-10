# =============================================================================
# Data Pipeline - Abstract Base Class for Data Providers
# =============================================================================
"""
Abstract base class defining the interface for all data providers.
This enables hot-swappable data sources with a consistent API.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import pandas as pd


class DataProviderBase(ABC):
    """
    Abstract base class for forex data providers.
    
    All data providers must implement these methods to ensure
    consistent behavior across different data sources.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this data provider."""
        pass
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            symbol: Currency pair symbol (e.g., "EURUSD")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            start: Start datetime (optional if days is provided)
            end: End datetime (defaults to now)
            days: Number of days of history (alternative to start)
            
        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index: DatetimeIndex
            
        Raises:
            ValueError: If symbol is invalid or data unavailable
            ConnectionError: If API is unreachable
        """
        pass
    
    @abstractmethod
    def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of currency pair symbols
            interval: Timeframe
            start: Start datetime
            end: End datetime
            days: Number of days of history
            
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available forex symbols from this provider.
        
        Returns:
            List of symbol strings (e.g., ["EURUSD", "GBPUSD", ...])
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid for this provider.
        
        Args:
            symbol: Currency pair symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for this provider.
        
        Default implementation removes common separators.
        Override in subclasses for provider-specific formatting.
        
        Args:
            symbol: Input symbol (e.g., "EUR/USD", "EUR-USD", "EURUSD")
            
        Returns:
            Normalized symbol string
        """
        return symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame column names to standard format.
        
        Ensures consistent column names across all providers:
        - 'open', 'high', 'low', 'close', 'volume'
        
        Args:
            df: Raw DataFrame from provider
            
        Returns:
            DataFrame with standardized column names
        """
        # Lowercase all column names
        df.columns = df.columns.str.lower()
        
        # Common column name mappings
        column_mapping = {
            'adj close': 'close',
            'adj_close': 'close',
            'adjusted_close': 'close',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add volume column if missing (some forex feeds don't have it)
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # Select only the columns we need
        return df[['open', 'high', 'low', 'close', 'volume']]
