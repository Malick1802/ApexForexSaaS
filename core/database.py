# =============================================================================
# Signal Database Manager
# =============================================================================
"""
SQLite database manager for persisting forex signals.

Schema:
- signals: Stores generated AI signals
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SignalDatabase:
    """
    Manages persistence of trading signals.
    """
    
    def __init__(self, db_path: str = "signals.db"):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def _init_db(self):
        """Initialize database schema."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Signals table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,  -- BUY, SELL, WAIT
                    confidence REAL NOT NULL,
                    model_version TEXT,
                    status TEXT DEFAULT 'NEW', -- NEW, SENT, EXECUTED
                    price_at_signal REAL,
                    tp_price REAL,
                    sl_price REAL,
                    raw_probabilities TEXT,
                    outcome TEXT DEFAULT 'ACTIVE' -- ACTIVE, SUCCESS, FAIL
                )
                """)
                
                # Indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON signals(symbol)")
                
                # Migration: Add columns if they don't exist (SQLite doesn't support IF NOT EXISTS for columns)
                # We check by trying to select them, if fail, we alter
                try:
                    cursor.execute("SELECT tp_price FROM signals LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute("ALTER TABLE signals ADD COLUMN tp_price REAL")
                    cursor.execute("ALTER TABLE signals ADD COLUMN sl_price REAL")
                    logger.info("Migrated signals table: Added tp_price and sl_price columns")
                
                try:
                    cursor.execute("SELECT outcome FROM signals LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute("ALTER TABLE signals ADD COLUMN outcome TEXT DEFAULT 'ACTIVE'")
                    logger.info("Migrated signals table: Added outcome column")
                
                conn.commit()
                logger.debug(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def save_signal(self, data: Dict[str, Any]) -> int:
        """
        Save a new signal to the database.
        
        Args:
            data: Dictionary containing signal details
            
        Returns:
            ID of the inserted row
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, signal, confidence, 
                    model_version, status, price_at_signal, 
                    tp_price, sl_price, raw_probabilities, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('timestamp', datetime.now().isoformat()),
                    data['symbol'],
                    data['signal'],
                    data['confidence'],
                    data.get('model_version', 'v1'),
                    data.get('status', 'NEW'),
                    data.get('price_at_signal', 0.0),
                    data.get('tp_price'),
                    data.get('sl_price'),
                    str(data.get('raw_probabilities', [])),
                    data.get('outcome', 'ACTIVE' if data['signal'] != 'WAIT' else 'N/A')
                ))
                
                signal_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Saved signal {signal_id} for {data['symbol']}: {data['signal']}")
                return signal_id
                
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return -1

    def get_recent_signals(self, limit: int = 50, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get recent signals, optionally filtered by symbol.
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute("""
                    SELECT * FROM signals 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT ?
                    """, (symbol, limit))
                else:
                    cursor.execute("""
                    SELECT * FROM signals 
                    ORDER BY timestamp DESC LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return []

    def get_active_signals(self) -> List[Dict]:
        """Get all signals currently marked as ACTIVE."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM signals WHERE outcome = 'ACTIVE'")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to fetch active signals: {e}")
            return []

    def has_active_signal(self, symbol: str) -> bool:
        """Check if there's already an active BUY/SELL signal for this symbol."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM signals WHERE symbol = ? AND outcome = 'ACTIVE' AND signal IN ('BUY', 'SELL')",
                    (symbol,)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Failed to check active signal for {symbol}: {e}")
            return False

    def resolve_signals(self, price_map: Dict[str, float]) -> Dict[str, str]:
        """
        Check all ACTIVE BUY/SELL signals against current prices.
        Resolves as SUCCESS if TP hit, FAIL if SL hit.
        
        Args:
            price_map: Dict of {symbol: current_price}
            
        Returns:
            Dict of {symbol: outcome} for resolved signals
        """
        resolved = {}
        try:
            active = self.get_active_signals()
            for sig in active:
                symbol = sig['symbol']
                signal_type = sig['signal']
                tp = sig.get('tp_price')
                sl = sig.get('sl_price')
                current_price = price_map.get(symbol)

                if not current_price or not tp or not sl:
                    continue
                if signal_type not in ('BUY', 'SELL'):
                    continue

                outcome = None
                if signal_type == 'BUY':
                    if current_price >= tp:
                        outcome = 'SUCCESS'
                    elif current_price <= sl:
                        outcome = 'FAIL'
                elif signal_type == 'SELL':
                    if current_price <= tp:
                        outcome = 'SUCCESS'
                    elif current_price >= sl:
                        outcome = 'FAIL'

                if outcome:
                    self.update_signal_outcome(sig['id'], outcome)
                    resolved[symbol] = outcome
                    logger.info(
                        f"{'✅' if outcome == 'SUCCESS' else '❌'} {symbol} {signal_type} → {outcome} "
                        f"(Entry: {sig.get('price_at_signal', 0):.5f}, Current: {current_price:.5f}, "
                        f"TP: {tp:.5f}, SL: {sl:.5f})"
                    )
        except Exception as e:
            logger.error(f"Failed to resolve signals: {e}")
        return resolved

    def expire_stale_signals(self, max_age_hours: int = 48):
        """Expire active signals older than max_age_hours as 'EXPIRED'."""
        try:
            cutoff = (datetime.now() - __import__('datetime').timedelta(hours=max_age_hours)).isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE signals SET outcome = 'EXPIRED' WHERE outcome = 'ACTIVE' AND timestamp < ?",
                    (cutoff,)
                )
                count = cursor.rowcount
                conn.commit()
                if count > 0:
                    logger.info(f"Expired {count} stale signals older than {max_age_hours}h")
        except Exception as e:
            logger.error(f"Failed to expire stale signals: {e}")

    def update_signal_outcome(self, signal_id: int, outcome: str):
        """Update the outcome (SUCCESS/FAIL/EXPIRED) of a specific signal."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE signals SET outcome = ? WHERE id = ?", (outcome, signal_id))
                conn.commit()
                logger.info(f"Signal {signal_id} marked as {outcome}")
        except Exception as e:
            logger.error(f"Failed to update signal outcome: {e}")
            
    def get_todays_stats(self) -> Dict[str, int]:
        """Get signal statistics for today."""
        try:
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT signal, COUNT(*) as count 
                FROM signals 
                WHERE timestamp >= ? 
                GROUP BY signal
                """, (start_of_day,))
                
                return {row[0]: row[1] for row in cursor.fetchall()}
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def clear_signals(self):
        """Clear all signals from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM signals")
                conn.commit()
                logger.info("All signals cleared from database.")
                return True
        except Exception as e:
            logger.error(f"Failed to clear signals: {e}")
            return False

