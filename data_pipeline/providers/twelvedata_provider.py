# =============================================================================
# TwelveData Provider (Stub for Future Implementation)
# =============================================================================
"""
TwelveData implementation of the DataProviderBase.
Commercial API for production-grade real-time and historical data.

This is a stub implementation - ready to be completed when
switching to the commercial API.
"""

import os
from datetime import datetime
from typing import List, Optional
import logging

import pandas as pd
import time
from threading import Lock

from data_pipeline.base import DataProviderBase

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls - 8 requests per minute (Free Tier)."""
    
    def __init__(self, max_requests: int = 8, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Check if we're at the limit
            if len(self.requests) >= self.max_requests:
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request) + 1.0 # 1s buffer
                if wait_time > 0:
                    logger.info(f"TwelveData Rate Limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests 
                                   if now - req_time < self.time_window]
            
            # Record this request
            self.requests.append(time.time())


class TwelveDataProvider(DataProviderBase):
    """
    Data provider using TwelveData API.
    
    Requires TWELVEDATA_API_KEY environment variable.
    
    Pros:
        - Real-time data available
        - Extensive forex coverage
        - High-quality historical data
        - Websocket support for streaming
        
    Cons:
        - Paid API (free tier has limits)
        - Rate limits apply
        
    Setup:
        1. Get API key from https://twelvedata.com/
        2. Set TWELVEDATA_API_KEY in .env file
    """
    
    # Interval mapping for TwelveData
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min", 
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
        "1wk": "1week",
        "1mo": "1month",
    }
    
    # Shared rate limiter across all instances in the same process
    _limiter = RateLimiter(max_requests=8, time_window=60)
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize TwelveData provider.
        
        Args:
            api_key: API key (defaults to TWELVEDATA_API_KEY env var)
            rate_limit_delay: Seconds between API calls
        """
        self._api_key = api_key or os.getenv("TWELVEDATA_API_KEY")
        self._rate_limit_delay = rate_limit_delay
        self._base_url = "https://api.twelvedata.com"
        
        if not self._api_key:
            logger.warning(
                "TWELVEDATA_API_KEY not set. "
                "TwelveData provider will not work without an API key."
            )
    
    @property
    def name(self) -> str:
        return "twelvedata"
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from TwelveData.
        """
        try:
            from twelvedata import TDClient
        except ImportError:
            raise ImportError("twelvedata library not installed. Run `pip install twelvedata`.")

        td = TDClient(apikey=self._api_key)
        
        # Determine outputsize based on days/interval roughly (TD limit is 5000)
        # Default roughly 500 candles if not specified
        outputsize = 5000
        
        # Normalize symbol (TD uses format like EUR/USD)
        # But wait, input symbol is usually "EURUSD". TD accepts "EUR/USD" usually.
        # Let's try inserting the slash if missing and len is 6
        norm_symbol = symbol
        if len(symbol) == 6 and symbol.isalpha() and "/" not in symbol:
            norm_symbol = f"{symbol[:3]}/{symbol[3:]}"
            
        logger.info(f"TwelveData Fetch: {norm_symbol} {interval}")
        
        # Enforce rate limit
        self._limiter.wait_if_needed()
        
        # Manual retry loop for DNS/Network errors
        max_retries = 3
        retry_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                ts = td.time_series(
                    symbol=norm_symbol,
                    interval=self.INTERVAL_MAP.get(interval, interval),
                    outputsize=outputsize,
                    start_date=start,
                    end_date=end
                )
                df = ts.as_pandas()
                
                if df is None or df.empty:
                   raise ValueError(f"No data returned for {symbol}")

                # Normalize DataFrame columns to standard lowercase
                df.columns = [c.lower() for c in df.columns]
                
                # Sort ascending (oldest first)
                df = df.sort_index(ascending=True)
                
                return df

            except Exception as e:
                # If it's a DNS/Connection error, retry
                is_network_err = "NameResolutionError" in str(e) or "ConnectionError" in str(e) or "addrinfo" in str(e)
                
                if is_network_err and attempt < max_retries - 1:
                    logger.warning(f"TwelveData Network/DNS Error for {symbol} (Attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s... {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                    continue
                    
                logger.error(f"TwelveData Error for {symbol}: {e}")
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
        Fetch multiple using loop (TD has batch but loop is safer for simple implementation).
        """
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.fetch_ohlcv(sym, interval, start, end, days)
            except Exception:
                continue
        return results

    def get_available_symbols(self) -> List[str]:
        return [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF", "AUDJPY",
            "GBPAUD", "GBPCAD", "GBPCHF", "AUDCAD", "AUDNZD", "CADJPY", "CHFJPY",
            "NZDJPY", "CADCHF", "NZDCAD", "NZDCHF", "EURNZD", "GBPNZD", "AUDCHF",
            "EURSGD", "USDSGD", "USDHKD"
        ]

    def validate_symbol(self, symbol: str) -> bool:
        return True # lax validation
