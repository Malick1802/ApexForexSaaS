# =============================================================================
# Signal Database Manager
# =============================================================================
"""
SQLite database manager for persisting forex signals.

Schema:
- signals: Stores generated AI signals
- audit_logs: Stores platform interactions and system events
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# ── ROOT DIRECTORY RESOLUTION ────────────────────────────────
# Finding the root ApexForexSaaS folder from the core/database.py file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "signals.db")

class SignalDatabase:
    """
    Manages persistence of trading signals.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        # Force absolute pathing to ensure all VM processes (Sentinel, Dashboard, Brain) sync
        if not db_path:
            db_path = DEFAULT_DB_PATH
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def get_signal_count(self) -> int:
        """Returns the total number of signals in the database."""
        try:
            with self._get_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
                return int(count)
        except Exception:
            return 0
        
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30) # Increase timeout for server concurrency
        # Enable WAL mode for better performance/concurrency on Azure Linux
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
        
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
                    tp_pips INTEGER,
                    sl_pips INTEGER,
                    model_trades INTEGER,
                    raw_probabilities TEXT,
                    outcome TEXT DEFAULT 'ACTIVE' -- ACTIVE, SUCCESS, FAIL
                )
                """)

                # Audit Logs table (NEW)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    level TEXT DEFAULT 'INFO',
                    details TEXT
                )
                """)
                
                # Indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON signals(symbol)")
                
                # Migration: Add columns if they don't exist
                columns_to_add = {
                    'tp_price': 'REAL',
                    'sl_price': 'REAL',
                    'tp_pips': 'INTEGER',
                    'sl_pips': 'INTEGER',
                    'model_trades': 'INTEGER',
                    'outcome': "TEXT DEFAULT 'ACTIVE'",
                    'buy_prob': 'REAL',
                    'sell_prob': 'REAL',
                    'wait_prob': 'REAL',
                    'mt5_ticket': 'TEXT',
                    'vix_proxy': 'REAL',
                    'yield_slope': 'REAL',
                    'regime': 'TEXT',
                    'regime_threshold': 'REAL',
                    'adx': 'REAL',
                    'atr_zscore': 'REAL',
                    'is_proven': 'INTEGER DEFAULT 0',
                    'is_hidden': 'INTEGER DEFAULT 0',
                    'expert_signal': 'TEXT',
                    'confidence_tier': 'INTEGER',
                    'buy_win_rate': 'REAL',
                    'sell_win_rate': 'REAL',
                    'suggested_lots': 'REAL',
                    'raw_confidence': 'REAL',
                    'expert_intent': 'TEXT'
                }
                
                cursor.execute("PRAGMA table_info(signals)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                
                for col, dtype in columns_to_add.items():
                    if col not in existing_cols:
                        try:
                            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col} {dtype}")
                            logger.info(f"Migrated signals table: Added {col} column")
                        except Exception as e:
                            logger.error(f"Migration failed for {col}: {e}")
                
                conn.commit()
                logger.debug(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def log_event(self, action: str, level: str = "INFO", details: Optional[str] = None):
        """
        Record a platform or system event in the audit_logs table.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO audit_logs (timestamp, action, level, details) VALUES (?, ?, ?, ?)",
                    (now, action, level, details)
                )
                conn.commit()
                logger.debug(f"Audit log: [{level}] {action}")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

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
                    timestamp, symbol, signal, expert_signal, confidence, confidence_tier,
                    model_version, status, price_at_signal, 
                    tp_price, sl_price, tp_pips, sl_pips, model_trades,
                    raw_probabilities, outcome, regime, vix_proxy,
                    yield_slope, buy_prob, sell_prob, wait_prob,
                    suggested_lots, is_proven, is_hidden, adx, atr_zscore,
                    raw_confidence, expert_intent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['timestamp'], 
                    data['symbol'], 
                    data['signal'], 
                    data.get('expert_signal', data['signal']),
                    data['confidence'],
                    data.get('confidence_tier', 0),
                    data.get('model_version', 'v1'),
                    data.get('status', 'NEW'),
                    data.get('price_at_signal', 0.0),
                    data.get('tp_price'),
                    data.get('sl_price'),
                    data.get('tp_pips'),
                    data.get('sl_pips'),
                    data.get('model_trades'),
                    str(data.get('raw_probabilities', [])),
                    data.get('outcome', 'ACTIVE'),
                    data.get('regime'),
                    data.get('vix_proxy'),
                    data.get('yield_slope'),
                    data.get('buy_prob'),
                    data.get('sell_prob'),
                    data.get('wait_prob'),
                    data.get('suggested_lots'),
                    data.get('is_proven', 0),
                    data.get('is_hidden', 0),
                    data.get('adx'),
                    data.get('atr_zscore'),
                    data.get('raw_confidence', 0.0),
                    data.get('expert_intent')
                ))

                
                signal_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Saved signal {signal_id} for {data['symbol']}: {data['signal']}")
                return signal_id
                
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return -1

    def save_heartbeat(self):
        """Saves a hidden system heartbeat to keep the Sentinel status active."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO signals (timestamp, symbol, signal, confidence, status, outcome)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.now(timezone.utc).isoformat(), 'SYSTEM', 'HEARTBEAT', 1.0, 'EXECUTED', 'SUCCESS'))
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

    def get_signal_by_id(self, signal_id: int) -> Optional[Dict]:
        """Fetch a single signal by its unique ID."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to fetch signal {signal_id}: {e}")
            return None

    def get_recent_signals(self, limit: int = 50, symbol: Optional[str] = None, include_hidden: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """Get recent signals, optionally filtered by symbol and hidden status."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                sql = "SELECT * FROM signals "
                params = []
                where_clauses = []
                
                # Filter by hidden status
                if not include_hidden:
                    where_clauses.append("(is_hidden IS NULL OR is_hidden = 0)")
                
                # Filter by symbol
                if symbol:
                    where_clauses.append("symbol = ?")
                    params.append(symbol)
                
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
                
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return []
                
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return []

    def get_active_signals(self, symbol: Optional[str] = None, include_hidden: bool = False) -> List[Dict]:
        """Get all signals currently marked as ACTIVE, optionally filtered by symbol and hidden status."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                sql = "SELECT * FROM signals WHERE outcome IN ('ACTIVE', 'N/A')"
                params = []
                
                if not include_hidden:
                    sql += " AND (is_hidden IS NULL OR is_hidden = 0)"
                
                if symbol:
                    sql += " AND symbol = ?"
                    params.append(symbol)
                    
                sql += " ORDER BY timestamp DESC"
                cursor.execute(sql, tuple(params))
                    
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    # Parse probabilities for frontend
                    try:
                        import json
                        raw = d.get('raw_probabilities', '[]')
                        if raw and raw != '[]':
                            probs = json.loads(raw)
                            if len(probs) == 3:
                                d['wait_prob'] = probs[0]
                                d['buy_prob'] = probs[1]
                                d['sell_prob'] = probs[2]
                    except:
                        pass
                    results.append(d)
                return results
        except Exception as e:
            logger.error(f"Failed to fetch active signals: {e}")
            return []

    def update_signal_metadata(self, signal_id: int, updates: dict):
        """Update specific columns of an existing signal (e.g., regime, confidence)."""
        if not updates:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Filter out keys that aren't valid columns for security
                valid_cols = [
                    'regime', 'raw_confidence', 'expert_intent', 'atr_zscore', 
                    'adx', 'yield_slope', 'buy_prob', 'sell_prob', 'wait_prob',
                    'outcome', 'status'
                ]
                
                set_clauses = []
                params = []
                for k, v in updates.items():
                    if k in valid_cols:
                        set_clauses.append(f"{k} = ?")
                        params.append(v)
                
                if not set_clauses:
                    return
                
                params.append(signal_id)
                sql = f"UPDATE signals SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(sql, tuple(params))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update signal {signal_id} metadata: {e}")

    def has_active_signal(self, symbol: str, include_hidden: bool = False) -> bool:
        """Check if there's already an active BUY/SELL signal for this symbol."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = "SELECT COUNT(*) FROM signals WHERE symbol = ? AND outcome = 'ACTIVE' AND signal IN ('BUY', 'SELL')"
                params = [symbol]
                
                if not include_hidden:
                    sql += " AND (is_hidden IS NULL OR is_hidden = 0)"
                
                cursor.execute(sql, tuple(params))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Failed to check active signal for {symbol}: {e}")
            return False

    def get_symbol_win_rates(self, symbol: str) -> Dict[str, float]:
        """
        Calculate realized win rates from the database history for this symbol.
        Formula: SUCCESS / (SUCCESS + FAIL)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # BUY stats
                cursor.execute(
                    "SELECT outcome, COUNT(*) FROM signals WHERE symbol = ? AND signal = 'BUY' AND outcome IN ('SUCCESS', 'FAIL') GROUP BY outcome",
                    (symbol,)
                )
                buy_stats = dict(cursor.fetchall())
                
                # SELL stats
                cursor.execute(
                    "SELECT outcome, COUNT(*) FROM signals WHERE symbol = ? AND signal = 'SELL' AND outcome IN ('SUCCESS', 'FAIL') GROUP BY outcome",
                    (symbol,)
                )
                sell_stats = dict(cursor.fetchall())
                
                def calc_win_rate(stats):
                    success = stats.get('SUCCESS', 0)
                    fail = stats.get('FAIL', 0)
                    total = success + fail
                    return (success / total * 100) if total > 0 else 0.0
                
                return {
                    'buy_win_rate': calc_win_rate(buy_stats),
                    'sell_win_rate': calc_win_rate(sell_stats)
                }
        except Exception as e:
            logger.error(f"Failed to calculate win rates for {symbol}: {e}")
            return {'buy_win_rate': 0.0, 'sell_win_rate': 0.0}

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

                # Safety Flush: Clear out corrupted legacy trades missing proper limits
                if tp is None or sl is None or tp == 0.0 or sl == 0.0:
                    logger.warning(f"EXPIRED: {symbol} (ID {sig['id']}) has missing TP/SL. Flushed to prevent deadlock.")
                    self.update_signal_outcome(sig['id'], 'EXPIRED')
                    resolved[symbol] = 'EXPIRED'
                    continue
                
                price_data = price_map.get(symbol)
                if not price_data:
                    continue
                if signal_type not in ('BUY', 'SELL'):
                    continue

                # Support both flat floats (legacy) and detailed price dicts
                if isinstance(price_data, dict):
                    high = price_data.get('high')
                    low = price_data.get('low')
                else:
                    high = price_data
                    low = price_data

                outcome = None
                if signal_type == 'BUY':
                    if high >= tp:
                        outcome = 'SUCCESS'
                    elif low <= sl:
                        outcome = 'FAIL'
                elif signal_type == 'SELL':
                    if low <= tp:
                        outcome = 'SUCCESS'
                    elif high >= sl:
                        outcome = 'FAIL'

                if outcome:
                    self.update_signal_outcome(sig['id'], outcome)
                    resolved[symbol] = outcome
                    logger.info(
                        f"{'✅' if outcome == 'SUCCESS' else '❌'} {symbol} {signal_type} → {outcome} "
                        f"(Entry: {sig.get('price_at_signal', 0):.5f}, TP: {tp:.5f}, SL: {sl:.5f})"
                    )
        except Exception as e:
            logger.error(f"Failed to resolve signals: {e}")
        return resolved

    def expire_stale_signals(self, max_age_hours: int = 48):
        """Expire active signals older than max_age_hours as 'EXPIRED'."""
        # DISABLED PERMANENTLY: User wants manual closure or TP/SL only.
        return
        try:
            cutoff = (datetime.now(timezone.utc) - __import__('datetime').timedelta(hours=max_age_hours)).isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # cursor.execute(
                #     "UPDATE signals SET outcome = 'EXPIRED' WHERE outcome = 'ACTIVE' AND timestamp < ?",
                #     (cutoff,)
                # )
                count = 0 # cursor.rowcount
                conn.commit()
                if count > 0:
                    logger.info(f"Expired {count} stale signals older than {max_age_hours}h")
        except Exception as e:
            logger.error(f"Failed to expire stale signals: {e}")

    def update_signal_status(self, signal_id: int, status: str):
        """Update the status (SENT, EXECUTED) of a specific signal."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE signals SET status = ? WHERE id = ?", (status, signal_id))
                conn.commit()
                logger.info(f"Signal {signal_id} status updated to {status}")
        except Exception as e:
            logger.error(f"Failed to update signal status: {e}")

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
            start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
            
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

    def get_performance_matrix_stats(self, lookback_days: int = 14) -> List[Dict]:
        """
        Aggregate performance metrics for all symbols within a lookback window.
        """
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN outcome = 'SUCCESS' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'FAIL' THEN 1 ELSE 0 END) as losses,
                        AVG(confidence) as avg_confidence,
                        MAX(timestamp) as last_trade
                    FROM signals
                    WHERE timestamp >= ? AND outcome IN ('SUCCESS', 'FAIL')
                    GROUP BY symbol
                    ORDER BY wins DESC
                """, (cutoff,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to fetch performance matrix stats: {e}")
            return []

    def get_model_registry_stats(self) -> List[Dict]:
        """
        Aggregate all-time performance metrics per symbol (model).
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as all_time_trades,
                        SUM(CASE WHEN outcome = 'SUCCESS' THEN 1 ELSE 0 END) as all_time_wins,
                        AVG(confidence) as all_time_confidence,
                        MAX(timestamp) as last_seen
                    FROM signals
                    WHERE outcome IN ('SUCCESS', 'FAIL')
                    GROUP BY symbol
                    ORDER BY symbol ASC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to fetch model registry stats: {e}")
            return []

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
