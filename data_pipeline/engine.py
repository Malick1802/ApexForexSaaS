# =============================================================================
# Data Pipeline Engine
# =============================================================================
"""
Main data engine that orchestrates data fetching and processing.

The engine uses a factory pattern for hot-swappable data providers,
allowing easy switching between yfinance (development) and
commercial APIs (production) via configuration.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Any

import yaml
import pandas as pd

from data_pipeline.base import DataProviderBase
from data_pipeline.providers.yfinance_provider import YFinanceProvider
from data_pipeline.providers.twelvedata_provider import TwelveDataProvider
from data_pipeline.labeling import triple_barrier_label, get_pip_value


logger = logging.getLogger(__name__)


# Provider registry for hot-swapping
PROVIDER_REGISTRY: Dict[str, Type[DataProviderBase]] = {
    "yfinance": YFinanceProvider,
    "twelvedata": TwelveDataProvider,
}


class DataEngine:
    """
    Main data engine for fetching and processing forex data.
    
    Features:
    - Hot-swappable data providers (yfinance, twelvedata, etc.)
    - Configuration-driven (reads from config.yaml)
    - Built-in labeling support
    - Optional local caching
    
    Usage:
        engine = DataEngine()
        df = engine.fetch("EURUSD", interval="1h", days=30)
        labeled_df = engine.fetch_labeled("EURUSD", interval="1h", days=30)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        provider_name: Optional[str] = None,
        cache_enabled: Optional[bool] = None
    ):
        """
        Initialize the data engine.
        
        Args:
            config_path: Path to config.yaml (defaults to project root)
            provider_name: Override the provider from config
            cache_enabled: Override cache setting from config
        """
        # Find project root (where config.yaml should be)
        self._project_root = self._find_project_root()
        
        # Load configuration
        config_path = config_path or os.path.join(self._project_root, "config.yaml")
        self._config = self._load_config(config_path)
        
        # Initialize provider
        active_provider = provider_name or self._config["data_provider"]["active"]
        self._provider = self._create_provider(active_provider)
        
        # Cache settings
        cache_config = self._config.get("data", {}).get("cache", {})
        self._cache_enabled = cache_enabled if cache_enabled is not None else cache_config.get("enabled", False)
        self._cache_dir = os.path.join(self._project_root, cache_config.get("directory", "data_cache"))
        self._cache_expiry_hours = cache_config.get("expiry_hours", 24)
        
        if self._cache_enabled:
            os.makedirs(self._cache_dir, exist_ok=True)
            
        # Pre-load currency pairs configuration
        self._pairs_config = self._build_pairs_lookup()
        
        logger.info(f"DataEngine initialized with provider: {self._provider.name}")
    
    def _find_project_root(self) -> str:
        """Find the project root directory."""
        # Start from current file and go up until we find config.yaml
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "config.yaml").exists():
                return str(current)
            current = current.parent
        
        # Fallback to current working directory
        return os.getcwd()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "data_provider": {
                "active": "yfinance",
                "yfinance": {"rate_limit_delay": 0.5},
                "twelvedata": {"rate_limit_delay": 1.0},
            },
            "trading": {
                "stop_loss_pips": 25,
                "take_profit_pips": 50,
                "pip_values": {"standard": 0.0001, "jpy": 0.01},
            },
            "currency_pairs": {"majors": [], "minors": [], "crosses": []},
            "data": {
                "default_history_days": 365,
                "cache": {"enabled": False},
            },
        }
    
    def _create_provider(self, provider_name: str) -> DataProviderBase:
        """Create a data provider instance."""
        if provider_name not in PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(PROVIDER_REGISTRY.keys())}"
            )
        
        provider_class = PROVIDER_REGISTRY[provider_name]
        provider_config = self._config["data_provider"].get(provider_name, {})
        
        # Extract relevant kwargs for the provider
        if provider_name == "yfinance":
            return provider_class(
                rate_limit_delay=provider_config.get("rate_limit_delay", 0.5)
            )
        elif provider_name == "twelvedata":
            return provider_class(
                api_key=provider_config.get("api_key"),
                rate_limit_delay=provider_config.get("rate_limit_delay", 1.0)
            )
        else:
            return provider_class()
    
    def _build_pairs_lookup(self) -> Dict[str, dict]:
        """Build a lookup dictionary for currency pair configurations."""
        lookup = {}
        pairs_config = self._config.get("currency_pairs", {})
        
        for category in ["majors", "minors", "crosses"]:
            for pair in pairs_config.get(category, []):
                symbol = pair.get("symbol", "")
                if symbol:
                    lookup[symbol] = {
                        "category": category,
                        "pip_type": pair.get("pip_type", "standard"),
                        "correlated_assets": pair.get("correlated_assets", []),
                    }
        
        return lookup
    
    @property
    def provider(self) -> DataProviderBase:
        """Get the current data provider."""
        return self._provider
    
    @property
    def config(self) -> dict:
        """Get the loaded configuration."""
        return self._config
    
    def switch_provider(self, provider_name: str) -> None:
        """
        Hot-swap to a different data provider.
        
        Args:
            provider_name: Name of provider ("yfinance", "twelvedata", etc.)
        """
        self._provider = self._create_provider(provider_name)
        logger.info(f"Switched to provider: {provider_name}")
    
    def get_pair_config(self, symbol: str) -> Optional[dict]:
        """
        Get configuration for a currency pair.
        
        Args:
            symbol: Currency pair symbol (e.g., "EURUSD")
            
        Returns:
            Configuration dict or None if not found
        """
        normalized = symbol.upper().replace("/", "").replace("-", "")
        return self._pairs_config.get(normalized)
    
    def get_pip_value(self, symbol: str) -> float:
        """
        Get the pip value for a currency pair.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Pip value (0.0001 for standard, 0.01 for JPY pairs)
        """
        pair_config = self.get_pair_config(symbol)
        if pair_config:
            pip_type = pair_config.get("pip_type", "standard")
            pip_values = self._config["trading"]["pip_values"]
            return pip_values.get(pip_type, 0.0001)
        
        # Fallback to auto-detection
        return get_pip_value(symbol)
    
    def get_correlated_assets(self, symbol: str) -> List[dict]:
        """
        Get correlated assets for a currency pair.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            List of correlated asset configs
        """
        pair_config = self.get_pair_config(symbol)
        return pair_config.get("correlated_assets", []) if pair_config else []
    
    def get_all_pairs(self, category: Optional[str] = None) -> List[str]:
        """
        Get all configured currency pairs.
        
        Args:
            category: Optional filter ("majors", "minors", "crosses")
            
        Returns:
            List of symbol strings
        """
        if category:
            return [
                symbol for symbol, config in self._pairs_config.items()
                if config["category"] == category
            ]
        return list(self._pairs_config.keys())
    
    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get the cache file path for a symbol/interval."""
        return Path(self._cache_dir) / f"{symbol}_{interval}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False
        
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        return age_hours < self._cache_expiry_hours
    
    def fetch(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None,
        use_cache: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a currency pair.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            interval: Timeframe ("1h", "4h", "1d", etc.)
            start: Start datetime
            end: End datetime
            days: Days of history (alternative to start)
            use_cache: Override cache setting
            
        Returns:
            DataFrame with OHLCV data
        """
        # Determine if we should use cache
        should_cache = use_cache if use_cache is not None else self._cache_enabled
        
        if should_cache:
            cache_path = self._get_cache_path(symbol, interval)
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading {symbol} from cache")
                return pd.read_parquet(cache_path)
        
        # Fetch from provider
        if days is None and start is None:
            days = self._config["data"].get("default_history_days", 365)
        
        df = self._provider.fetch_ohlcv(symbol, interval, start, end, days)
        
        # Cache the result
        if should_cache:
            df.to_parquet(cache_path)
            logger.info(f"Cached {symbol} to {cache_path}")
        
        return df
    
    def fetch_multiple(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = "1h",
        days: Optional[int] = None,
        category: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols (or None to use all configured)
            interval: Timeframe
            days: Days of history
            category: Filter by category if symbols is None
            
        Returns:
            Dict mapping symbol -> DataFrame
        """
        if symbols is None:
            symbols = self.get_all_pairs(category)
        
        return self._provider.fetch_multiple(symbols, interval, days=days)
    
    def fetch_labeled(
        self,
        symbol: str,
        interval: str = "1h",
        days: Optional[int] = None,
        stop_loss_pips: Optional[float] = None,
        take_profit_pips: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Fetch data and apply Triple Barrier labeling.
        
        Args:
            symbol: Currency pair
            interval: Timeframe
            days: Days of history
            stop_loss_pips: Override SL from config
            take_profit_pips: Override TP from config
            
        Returns:
            DataFrame with OHLCV data and 'label' column
        """
        # Get data
        df = self.fetch(symbol, interval, days=days)
        
        # Get labeling parameters
        trading_config = self._config["trading"]
        sl_pips = stop_loss_pips or trading_config["stop_loss_pips"]
        tp_pips = take_profit_pips or trading_config["take_profit_pips"]
        pip_value = self.get_pip_value(symbol)
        
        # Apply labeling
        labeled_df = triple_barrier_label(
            df,
            stop_loss_pips=sl_pips,
            take_profit_pips=tp_pips,
            pip_value=pip_value,
            symbol=symbol
        )
        
        return labeled_df
    
    def fetch_with_correlations(
        self,
        symbol: str,
        interval: str = "1h",
        days: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for a symbol and its correlated assets.
        
        Args:
            symbol: Primary currency pair
            interval: Timeframe
            days: Days of history
            
        Returns:
            Dict with 'primary' and correlated asset DataFrames
        """
        result = {"primary": self.fetch(symbol, interval, days=days)}
        
        correlated = self.get_correlated_assets(symbol)
        for asset in correlated:
            asset_symbol = asset["symbol"]
            try:
                # Correlated assets might be stocks/commodities, not forex
                # yfinance handles these without the =X suffix
                result[asset_symbol] = self._provider.fetch_ohlcv(
                    asset_symbol.replace("=X", ""),  # Remove forex suffix if present
                    interval,
                    days=days
                )
            except Exception as e:
                logger.warning(f"Failed to fetch correlated asset {asset_symbol}: {e}")
        
        return result


def create_engine(
    provider: str = "yfinance",
    config_path: Optional[str] = None
) -> DataEngine:
    """
    Factory function to create a DataEngine.
    
    Args:
        provider: Data provider name
        config_path: Optional path to config file
        
    Returns:
        Configured DataEngine instance
    """
    return DataEngine(config_path=config_path, provider_name=provider)
