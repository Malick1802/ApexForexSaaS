# =============================================================================
# YFinance Data Provider
# =============================================================================
"""
YFinance implementation of the DataProviderBase.
Free data source suitable for development and backtesting.
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional
import logging

import pandas as pd
import yfinance as yf

from data_pipeline.base import DataProviderBase


logger = logging.getLogger(__name__)


class YFinanceProvider(DataProviderBase):
    """
    Data provider using Yahoo Finance (yfinance library).
    
    Pros:
        - Free, no API key required
        - Good historical data coverage
        - Reliable for daily/hourly timeframes
        
    Cons:
        - No real-time data
        - Limited to certain forex pairs
        - Rate limiting can be aggressive
        
    Note: Forex symbols in yfinance use the =X suffix (e.g., EURUSD=X)
    """
    
    # Valid timeframe mappings for yfinance
    INTERVAL_MAP = {
        "1m": "1m",
        "2m": "2m", 
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "60m": "1h",
        "90m": "90m",
        "4h": "4h",  # Note: May need special handling
        "1d": "1d",
        "1D": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
    }
    
    # Known forex pairs available on Yahoo Finance
    FOREX_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF", "AUDJPY",
        "GBPAUD", "GBPCAD", "GBPCHF", "AUDCAD", "AUDNZD", "CADJPY", "CHFJPY",
        "NZDJPY", "CADCHF", "NZDCAD", "NZDCHF", "EURNZD", "GBPNZD", "AUDCHF",
        "EURSGD", "USDSGD", "USDHKD",
    ]
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize YFinance provider.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls
        """
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        
    @property
    def name(self) -> str:
        return "yfinance"
    
    def _to_yf_symbol(self, symbol: str) -> str:
        """Convert standard forex symbol to yfinance format."""
        # If symbol already has yfinance structure (e.g. GC=F, ^VIX, or ends with =X), use as is
        if "=" in symbol or "^" in symbol:
             return symbol
             
        normalized = self.normalize_symbol(symbol)
        return f"{normalized}=X"
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _get_period_for_interval(self, interval: str, days: int) -> str:
        """
        Get appropriate yfinance period string for the interval.
        
        yfinance has limitations on how far back you can fetch
        for intraday intervals.
        """
        if interval in ["1m", "2m"]:
            # Max 7 days for 1m/2m
            return f"{min(days, 7)}d"
        elif interval in ["5m", "15m", "30m"]:
            # Max 60 days for these intervals
            return f"{min(days, 60)}d"
        elif interval in ["1h", "60m", "90m"]:
            # Max 730 days for hourly
            return f"{min(days, 730)}d"
        else:
            # Daily or longer - use max available
            return "max" if days > 365 * 10 else f"{days}d"
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Forex pair (e.g., "EURUSD")
            interval: Timeframe (e.g., "1h", "1d")
            start: Start datetime
            end: End datetime (defaults to now)
            days: Days of history (alternative to start)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize inputs
        yf_symbol = self._to_yf_symbol(symbol)
        yf_interval = self.INTERVAL_MAP.get(interval, interval)
        
        if end is None:
            end = datetime.now()
            
        if start is None and days is not None:
            # Clamp days based on interval limitations
            if interval in ["1h", "60m", "90m"]:
                days_to_fetch = min(days, 729) # 730 is strict limit, go slightly under
            elif interval in ["5m", "15m", "30m"]:
                days_to_fetch = min(days, 59)
            elif interval in ["1m", "2m"]:
                days_to_fetch = min(days, 6)
            else:
                days_to_fetch = days
                
            start = end - timedelta(days=days_to_fetch)
            
            if days_to_fetch < days:
                logger.warning(
                     f"Requested {days} days for {interval} but capped at {days_to_fetch} "
                     "due to yfinance limitations."
                )
        elif start is None:
            start = end - timedelta(days=365)  # Default 1 year
            
        logger.info(f"Fetching {symbol} ({yf_symbol}) - {interval} from {start} to {end}")
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            # Fetch data
            df = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,
                actions=False
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
                
            # Normalize to standard format
            df = self.normalize_dataframe(df)
            
            # Ensure index is timezone-naive for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise
    
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
            symbols: List of forex pairs
            interval: Timeframe
            start: Start datetime
            end: End datetime  
            days: Days of history
            
        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, interval, start, end, days)
                results[symbol] = df
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")
                continue
                
        return results
    
    def get_available_symbols(self) -> List[str]:
        """Return list of known forex pairs."""
        return self.FOREX_PAIRS.copy()
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is a known forex pair."""
        normalized = self.normalize_symbol(symbol)
        return normalized in self.FOREX_PAIRS
