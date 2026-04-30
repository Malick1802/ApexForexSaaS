import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)

# Root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHITELIST_PATH = PROJECT_ROOT / "config" / "trading_whitelist.json"
DEFAULT_DB_PATH = str(PROJECT_ROOT / "signals.db")

DEFAULT_HURDLE = 0.60  # 60% win rate
MIN_TRADES = 3        # Must have at least 3 resolved trades to be 'proven'
TIERS = [50, 55, 60, 70, 80, 90, 100]

class PerformanceGate:
    """
    Multi-Tier Trading Whitelist Manager.
    Evaluates pairs independently by DIRECTION at 5 confidence hurdles (60, 70, 80, 90, 100).
    Condition: Realized Accuracy >= 70% AND Resolved Trades >= 2 per tier.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        # matrix: { symbol: { direction: { tier_str: { accuracy, trades, status } } } }
        self.performance_matrix: Dict[str, Dict[str, Dict[str, Dict]]] = {}
        self.last_disk_load = 0
        self.load_whitelist()

    def load_whitelist(self):
        """Load performance tiers from JSON with absolute path protection."""
        try:
            if WHITELIST_PATH.exists():
                with open(WHITELIST_PATH, "r") as f:
                    data = json.load(f)
                    self.performance_matrix = data.get("performance_matrix", {})
                    self.last_disk_load = os.path.getmtime(WHITELIST_PATH)
                    logger.info(f"Performance Matrix loaded from {WHITELIST_PATH}")
            else:
                logger.warning(f"Whitelist not found at {WHITELIST_PATH}. Using empty matrix.")
                self.performance_matrix = {}
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")

    def save_whitelist(self):
        """Persist performance matrix to disk."""
        WHITELIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "performance_matrix": self.performance_matrix
        }
        with open(WHITELIST_PATH, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Performance Matrix saved for {len(self.performance_matrix)} symbols.")

    def is_tier_approved(self, symbol: str, direction: str, confidence) -> bool:
        """Check if the specific confidence tier for a pair/direction is officially APPROVED."""
        return self.get_tier_status(symbol, direction, confidence) == "APPROVED"

    def _normalize_tier(self, confidence) -> int:
        """Handle both float (0.85) and formatted strings ('70%') / ints (85)."""
        if isinstance(confidence, str):
            try:
                conf_int = int(confidence.replace('%', ''))
            except:
                return 0
        else:
            conf_int = int(confidence) if confidence >= 1 else int(confidence * 100)
        
        applicable_tier = 0
        for t in reversed(TIERS):
            if conf_int >= t:
                applicable_tier = t
                break
        return applicable_tier

    def get_tier_status(self, symbol: str, direction: str, confidence) -> str:
        """
        Get the status (APPROVED/BENCHED) of a specific confidence tier for a pair and direction.
        Supports both float (0.70) and formatted strings ('70%').
        """
        # Auto-Reload Check: If file on disk is newer than our memory, reload it.
        try:
            if WHITELIST_PATH.exists():
                mtime = os.path.getmtime(WHITELIST_PATH)
                if mtime > self.last_disk_load:
                    self.load_whitelist()
        except:
            pass

        if symbol not in self.performance_matrix:
            return "⬜ No data"
            
        applicable_tier = self._normalize_tier(confidence)
        if not applicable_tier:
            return "⬜ No data"

        direction_data = self.performance_matrix[symbol].get(direction)
        
        # Fallback: If specific direction is missing or this tier isn't in it, check 'ALL'
        if not direction_data or str(applicable_tier) not in direction_data:
            if direction != "ALL":
                direction_data = self.performance_matrix[symbol].get("ALL")
        
        if not direction_data:
            return "⬜ No data"
            
        tier_data = direction_data.get(str(applicable_tier))
        if not tier_data:
            return "⬜ No data"
            
        return tier_data.get("status", "⬜ No data")

    def get_tier_accuracy(self, symbol: str, direction: str, confidence) -> float:
        """Get the historically validated accuracy/win rate for a specific tier."""
        # Ensure fresh data
        self.get_tier_status(symbol, direction, confidence)
        
        applicable_tier = self._normalize_tier(confidence)
        if not applicable_tier:
            return 0.0

        direction_data = self.performance_matrix.get(symbol, {}).get(direction)
        if not direction_data or str(applicable_tier) not in direction_data:
            if direction != "ALL":
                direction_data = self.performance_matrix.get(symbol, {}).get("ALL")
        
        if not direction_data:
            return 0.0
            
        tier_data = direction_data.get(str(applicable_tier))
        if not tier_data:
            return 0.0
            
        # Support both 'accuracy' (live) and 'win_rate' (OOS script)
        return float(tier_data.get("accuracy", tier_data.get("win_rate", 0.0)))

    def recompute_from_db(self, lookback_days: int = 14):
        """
        Scan signals.db for the last X days and update all 5 tiers for all symbols BY DIRECTION.
        Recency Rule applies perfectly matching user logic.
        """
        if not Path(self.db_path).exists():
            return

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        found_recent = False

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Get all symbols seen recently
            cursor.execute("SELECT DISTINCT symbol FROM signals WHERE timestamp >= ?", (cutoff_date,))
            symbols = [row[0] for row in cursor.fetchall()]

            for symbol in symbols:
                if symbol not in self.performance_matrix:
                    self.performance_matrix[symbol] = {"BUY": {}, "SELL": {}, "ALL": {}}
                
                # Ensure all directions exist
                for d in ["BUY", "SELL", "ALL"]:
                    if d not in self.performance_matrix[symbol]:
                        self.performance_matrix[symbol][d] = {}
                
                for direction in ["BUY", "SELL", "ALL"]:
                    for t in TIERS:
                        # 2. Calculate tier accuracy
                        # For 'ALL', we don't filter by signal type
                        if direction == "ALL":
                            query = """
                                SELECT outcome, COUNT(*) as count 
                                FROM signals 
                                WHERE symbol = ? 
                                AND timestamp >= ? 
                                AND confidence >= ?
                                AND outcome IN ('SUCCESS', 'FAIL')
                                GROUP BY outcome
                            """
                            params = (symbol, cutoff_date, t / 100.0)
                        else:
                            query = """
                                SELECT outcome, COUNT(*) as count 
                                FROM signals 
                                WHERE symbol = ? 
                                AND signal = ?
                                AND timestamp >= ? 
                                AND confidence >= ?
                                AND outcome IN ('SUCCESS', 'FAIL')
                                GROUP BY outcome
                            """
                            params = (symbol, direction, cutoff_date, t / 100.0)

                        cursor.execute(query, params)
                        
                        stats = {row['outcome']: row['count'] for row in cursor.fetchall()}
                        success = stats.get('SUCCESS', 0)
                        fail = stats.get('FAIL', 0)
                        total = success + fail
                        
                        existing_tier = self.performance_matrix.get(symbol, {}).get(direction, {}).get(str(t), {})
                        
                        oos_trades = existing_tier.get("oos_trades", 0)
                        oos_accuracy = existing_tier.get("oos_accuracy", 0.0)
                        oos_success = int(round(oos_trades * oos_accuracy))
                        
                        total_combined = total + oos_trades
                        success_combined = success + oos_success
                        
                        if total_combined == 0:
                            # If we previously had data for this tier but now have 0 trades combined,
                            # we should mark it as BENCHED/No Data instead of leaving it stale.
                            if direction in self.performance_matrix[symbol] and str(t) in self.performance_matrix[symbol][direction]:
                                del self.performance_matrix[symbol][direction][str(t)]
                            continue 
                        
                        found_recent = True
                        win_rate = (success_combined / total_combined) if total_combined > 0 else 0.0
                        
                        # Apply Strict Approval Logic
                        approved = (total_combined >= MIN_TRADES) and (win_rate >= DEFAULT_HURDLE)
                        
                        source_label = "OOS + Live" if (oos_trades > 0 and total > 0) else ("OOS Jumpstart (45d)" if oos_trades > 0 else f"Live Data ({lookback_days}d)")
                        
                        self.performance_matrix[symbol][direction][str(t)] = {
                            "accuracy": win_rate,
                            "win_rate": win_rate,
                            "trades": total_combined,
                            "oos_trades": oos_trades,
                            "oos_accuracy": oos_accuracy,
                            "status": "APPROVED" if approved else "BENCHED",
                            "last_updated": datetime.now(timezone.utc).isoformat(),
                            "source": source_label
                        }

            conn.close()
            if found_recent:
                 self.save_whitelist()
            
        except Exception as e:
            logger.error(f"Matrix recompute failed: {e}")

    def sync_with_audit_report(self, audit_json_path: str):
        """
        Import matrix results from the static OOS audit JSON.
        Required to establish the initial 'Proven' baseline across all tiers.
        """
        path = Path(audit_json_path)
        if not path.exists():
            logger.error(f"Audit JSON not found at {path}")
            return

        try:
            with open(path, "r") as f:
                audit_data = json.load(f)
            
            for symbol, directions in audit_data.items():
                if symbol not in self.performance_matrix:
                    self.performance_matrix[symbol] = {"BUY": {}, "SELL": {}}
                
                for direction, tiers in directions.items():
                    if direction not in self.performance_matrix[symbol]:
                        self.performance_matrix[symbol][direction] = {}
                        
                    for t_str, data in tiers.items():
                        acc = data.get("accuracy", 0.0)
                        trades = data.get("trades", 0)
                        
                        # Apply Strict Hurdle: >= 2 trades and >= 70% win rate
                        approved = (trades >= MIN_TRADES) and (acc >= DEFAULT_HURDLE)
                        
                        # Baseline Sync: Set or Update
                        self.performance_matrix[symbol][direction][t_str] = {
                            "accuracy": acc,
                            "trades": trades,
                            "status": "APPROVED" if approved else "BENCHED",
                            "last_updated": datetime.now(timezone.utc).isoformat(),
                            "source": "Audit Baseline"
                        }
            
            self.save_whitelist()
            logger.info("Multi-Tier Sync Complete.")
            
        except Exception as e:
            logger.error(f"Audit sync failed: {e}")

    def update_stats(self, symbol: str, direction: str, tier: int, outcome: str):
        """Update stats for a specific symbol/direction/tier based on live outcome directly."""
        if symbol not in self.performance_matrix:
            self.performance_matrix[symbol] = {"BUY": {}, "SELL": {}}
        if direction not in self.performance_matrix[symbol]:
            self.performance_matrix[symbol][direction] = {}
            
        t_str = str(tier)
        # 1. Ensure Tier Entry Exists
        if t_str not in self.performance_matrix[symbol][direction]:
            self.performance_matrix[symbol][direction][t_str] = {
                "accuracy": 0.0, "trades": 0, 
                "status": "BENCHED", "source": "Live Init", "successes": 0
            }
            
        data = self.performance_matrix[symbol][direction][t_str]
        
        # Retrofit successes if we only have accuracy/trades
        if "successes" not in data:
            data["successes"] = int(data.get("accuracy", 0) * data.get("trades", 0))

        # 3. Apply Outcome
        if outcome == "SUCCESS":
            data["successes"] += 1
            
        data["trades"] += 1
        data["accuracy"] = data["successes"] / data["trades"] if data["trades"] > 0 else 0.0
        
        # Re-evaluate status
        approved = (data["trades"] >= MIN_TRADES) and (data["accuracy"] >= DEFAULT_HURDLE)
        data["status"] = "APPROVED" if approved else "BENCHED"
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        data["source"] = "Live Incremental"
        
        logger.info(f"Stats Update [{symbol} {direction}@{tier}]: {outcome}. New Acc: {data['accuracy']:.1%} ({data['trades']} trades)")
        self.save_whitelist()

# Singleton helper
_gate = None
def get_performance_gate() -> PerformanceGate:
    global _gate
    if _gate is None:
        _gate = PerformanceGate()
    return _gate
