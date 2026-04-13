# =============================================================================
# MetaTrader 5 Data Provider
# =============================================================================
"""
MT5 implementation of the DataProviderBase.
Uses MetaTrader5 Python library to fetch historical data directly from the broker.

Advantages over YFinance:
- 5+ years of 1h data (vs Yahoo's 729 days)
- Broker's actual prices (what you trade on)
- No rate limits (data is local/server-cached)
- Includes tick volume
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pathlib import Path
import pandas as pd

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.core.mt5_connector import get_mt5

from data_pipeline.base import DataProviderBase

logger = logging.getLogger(__name__)


# Note: INTERVAL_MAP will now be populated at runtime to ensure the mt5 object is alive
INTERVAL_MAP = {
    "1m":  "TIMEFRAME_M1",
    "5m":  "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h":  "TIMEFRAME_H1",
    "4h":  "TIMEFRAME_H4",
    "1d":  "TIMEFRAME_D1",
    "1w":  "TIMEFRAME_W1",
}


class MT5Provider(DataProviderBase):
    """
    Data provider using MetaTrader 5 terminal.
    
    Requires:
        - MetaTrader 5 terminal running on the system
        - MetaTrader5 Python package installed
        - Active broker login
    """
    
    def __init__(self):
        """Initialize MT5 provider."""
        import threading
        self.thread_local = threading.local()
        # Initial connection check
        self._connect()
    
    def _connect(self):
        """Get the shared MT5 connection from the connector."""
        self.mt5 = get_mt5()
        if self.mt5 is None:
            raise ConnectionError(
                "Failed to connect to MT5 Bridge. "
                "Ensure the mt5linux bridge is running at 127.0.0.1:18812."
            )
        self.thread_local.connected = True
        
        account = self.mt5.account_info()
        if account:
            logger.info(
                f"MT5 connected on thread: {account.server}, "
                f"Account #{account.login}, "
                f"Balance: {account.balance}"
            )
        self.thread_local.connected = True
    
    @property
    def name(self) -> str:
        return "mt5"
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for MT5 broker.
        """
        self._connect()
        # 1. Try raw symbol as-is
        info = self.mt5.symbol_info(symbol)
        if info is not None:
            return symbol

        # 2. Try cleaned uppercase version
        clean = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
        info = self.mt5.symbol_info(clean)
        if info is not None:
            return clean
        
        # 3. Try common broker suffixes
        for suffix in ["m", ".raw", ".ecn", ".i", "_i", "pro", ".pro"]:
            test = clean + suffix
            info = self.mt5.symbol_info(test)
            if info is not None:
                logger.info(f"Symbol mapped: {symbol} -> {test}")
                return test
        
        # Return upper version as backup
        return clean
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from MetaTrader 5.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            interval: Timeframe (e.g., "1h", "4h", "1d")
            start: Start datetime (optional)
            end: End datetime (defaults to now)
            days: Number of days of history
            
        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index: DatetimeIndex (UTC)
        """
        self._connect()
        
        mt5_symbol = self.normalize_symbol(symbol)
        
        # Get constant from the mt5 instance
        tf_name = INTERVAL_MAP.get(interval)
        timeframe = getattr(self.mt5, tf_name) if tf_name else None
        
        if timeframe is None:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Ensure symbol is available for data
        if not self.mt5.symbol_select(mt5_symbol, True):
            raise ValueError(f"Cannot select symbol {mt5_symbol} in MT5 (Thread Context Issue)")
        
        # Determine date range
        if end is None:
            end = datetime.now(timezone.utc)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
            
        if start is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        elif days is not None:
            start = end - timedelta(days=days)
        else:
            start = end - timedelta(days=365)
        
        logger.info(
            f"Fetching {mt5_symbol} {interval} from {start.date()} to {end.date()} "
            f"(~{(end - start).days} days)"
        )
        
        # Fetch rates from MT5
        rates = self.mt5.copy_rates_range(mt5_symbol, timeframe, start, end)
        
        if rates is None or len(rates) == 0:
            error = self.mt5.last_error()
            raise ValueError(
                f"No data returned for {mt5_symbol} ({interval}): {error}"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # MT5 returns 'time' as Unix timestamp
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        # MT5 columns: time, open, high, low, close, tick_volume, spread, real_volume
        # Map to standard format
        df = df.rename(columns={
            'tick_volume': 'volume'
        })
        
        # Select standard columns
        result = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        logger.info(f"Fetched {len(result)} candles for {mt5_symbol}")
        return result
    
    def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> dict:
        """Fetch OHLCV data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(
                    symbol, interval=interval,
                    start=start, end=end, days=days
                )
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results
    
    def get_available_symbols(self) -> List[str]:
        """Get all forex symbols available in MT5."""
        self._connect()
        symbols = self.mt5.symbols_get()
        if symbols is None:
            return []
        
        # Filter for forex pairs (typically in "Forex" or "FX" group)
        forex = []
        for s in symbols:
            # Common group names for forex
            if any(x in s.path.lower() for x in ["forex", "fx", "currencies"]):
                forex.append(s.name)
        
        # If no group filtering works, return all
        if not forex:
            forex = [s.name for s in symbols]
            
        return sorted(forex)
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists in MT5."""
        self._connect()
        mt5_symbol = self.normalize_symbol(symbol)
        info = self.mt5.symbol_info(mt5_symbol)
        return info is not None
    
    def __del__(self):
        """Provider cleanup - singleton handling delegated to MT5Connector."""
        pass
