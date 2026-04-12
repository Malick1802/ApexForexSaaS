# =============================================================================
# MT5 Connector - Centralized Linux/Windows Logic
# =============================================================================
"""
Centralized MetaTrader 5 connection manager that handles both native Windows
and Linux/Wine bridge (via mt5linux). 
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Optional
import platform

# Setup Logging
logger = logging.getLogger(__name__)

# Intelligent Bridge/Native Selection
IS_WINDOWS = platform.system() == "Windows"

try:
    if IS_WINDOWS:
        import MetaTrader5 as mt5
        BRIDGE_MODE = False
        logger.info("🖥️ Operating in Native Windows MT5 mode.")
    else:
        from mt5linux import MetaTrader5 as mt5
        BRIDGE_MODE = True
        logger.info("🌉 Operating in Linux/Wine Bridge mode.")
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MT5 packages not found. Run: pip install MetaTrader5 mt5linux")

class MT5Connector:
    """
    Singleton connector for MT5.
    Handles persistent connection state for the entire application.
    """
    _instance: Optional['MT5Connector'] = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MT5Connector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization of instance vars
        if not hasattr(self, 'initialized'):
            self.config = self._load_config()
            self.initialized = True

    def _load_config(self) -> dict:
        """Find and load config.yaml."""
        # Find project root (robust anchor)
        current = Path(__file__).resolve().parent
        while current != current.parent:
            config_path = current / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f).get('mt5', {})
            current = current.parent
        return {}

    def get_connection(self):
        """
        Return the active MT5 connection object.
        Initializes the connection if not already active.
        """
        if not MT5_AVAILABLE:
            logger.error("🚫 MT5 Interface unavailable (MetaTrader5/mt5linux missing).")
            return None

        # If connection exists but is not responding, reset it
        if self._connection:
            try:
                # Simple check to see if connection is alive
                if self._connection.account_info() is not None:
                    return self._connection
            except Exception:
                logger.warning("🔄 MT5 connection stale. Re-initializing...")
                self._connection = None

        return self._initialize_connection()

    def _initialize_connection(self):
        """Perform the actual MT5 initialization (Native or Bridge)."""
        if BRIDGE_MODE:
            logger.info("🔗 Attempting link to Linux MT5 Bridge (Wine)...")
        else:
            logger.info("🔗 Attempting native Windows MT5 connection...")
        
        try:
            # For mt5linux (Bridge), we need to instantiate. 
            # For MetaTrader5 (Native), we use the module directly.
            conn = mt5() if BRIDGE_MODE else mt5
            
            # Read parameters from config
            login = self.config.get('login')
            password = self.config.get('password')
            server = self.config.get('server')

            if BRIDGE_MODE:
                host = self.config.get('host', '127.0.0.1')
                port = self.config.get('port', 18812)
                logger.info(f"Connecting to {host}:{port} (Account: {login})")
                success = conn.initialize(
                    host=host,
                    port=port,
                    login=login,
                    password=password,
                    server=server
                )
            else:
                logger.info(f"Connecting to native terminal (Account: {login})")
                success = conn.initialize(
                    login=login,
                    password=password,
                    server=server
                )

            if success:
                acc = conn.account_info()
                if acc:
                    logger.info(f"✅ MT5 CONNECTED: {acc.server} (Login: {acc.login})")
                self._connection = conn
                return conn
            else:
                err = conn.last_error()
                logger.error(f"❌ MT5 Initialization Failed: {err}")
                return None

        except Exception as e:
            logger.error(f"💥 Critical error connecting to MT5: {e}")
            return None

    def shutdown(self):
        """Gracefully release the connection."""
        if self._connection:
            try:
                self._connection.shutdown()
                logger.info("🔌 MT5 disconnected.")
            except Exception:
                pass
            self._connection = None

# Global helper for quick access
def get_mt5():
    """Returns the singleton MT5 connection object."""
    return MT5Connector().get_connection()
