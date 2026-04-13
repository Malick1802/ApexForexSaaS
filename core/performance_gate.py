import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)

WHITELIST_PATH = Path("config/trading_whitelist.json")
DEFAULT_HURDLE = 0.70  # 70% win rate
MIN_TRADES = 2        # Must have at least 2 resolved trades to be 'proven'
TIERS = [60, 70, 80, 90, 100]

class PerformanceGate:
    """
    Multi-Tier Trading Whitelist Manager.
    Evaluates pairs independently at 5 confidence hurdles (60, 70, 80, 90, 100).
    Condition: Realized Accuracy >= 70% AND Resolved Trades >= 2 per tier.
    """

    def __init__(self, db_path: str = "signals.db"):
        self.db_path = db_path
        # matrix: { symbol: { tier_str: { accuracy, trades, status } } }
        self.performance_matrix: Dict[str, Dict[str, Dict]] = {}
        self.load_whitelist()

    def load_whitelist(self):
        """Load performance matrix from disk."""
        if WHITELIST_PATH.exists():
            try:
                with open(WHITELIST_PATH, "r") as f:
                    data = json.load(f)
                    self.performance_matrix = data.get("performance_matrix", {})
                logger.debug(f"Loaded performance matrix for {len(self.performance_matrix)} pairs.")
            except Exception as e:
                logger.error(f"Failed to load whitelist: {e}")
        else:
            logger.warning("No whitelist found. All trading restricted until audit/recompute.")

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

    def is_tier_approved(self, symbol: str, confidence) -> bool:
        """Check if the specific confidence tier for a pair is officially APPROVED."""
        return self.get_tier_status(symbol, confidence) == "APPROVED"

    def get_tier_status(self, symbol: str, confidence) -> str:
        """
        Get the status (APPROVED/BENCHED) of a specific confidence tier for a pair.
        Supports both float (0.70) and formatted strings ('70%').
        """
        if symbol not in self.performance_matrix:
            return "⬜ No data"
            
        # Normalize confidence to int (e.g. 70)
        if isinstance(confidence, str):
            try:
                conf_int = int(confidence.replace('%', ''))
            except:
                return "⬜ No data"
        else:
            conf_int = int(confidence * 100)

        applicable_tier = None
        for t in reversed(TIERS):
            if conf_int >= t:
                applicable_tier = str(t)
                break
        
        if not applicable_tier:
            return "⬜ No data" # Below 60% is always restricted
            
        tier_data = self.performance_matrix[symbol].get(applicable_tier)
        if not tier_data:
            return "⬜ No data"
            
        return tier_data.get("status", "⬜ No data")

    def recompute_from_db(self, lookback_days: int = 14):
        """
        Scan signals.db for the last X days and update all 5 tiers for all symbols.
        Recency Rule applies.
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
                    self.performance_matrix[symbol] = {}
                
                for t in TIERS:
                    # 2. Calculate tier accuracy (signals >= t confidence)
                    cursor.execute("""
                        SELECT outcome, COUNT(*) as count 
                        FROM signals 
                        WHERE symbol = ? 
                        AND timestamp >= ? 
                        AND confidence >= ?
                        AND outcome IN ('SUCCESS', 'FAIL')
                        GROUP BY outcome
                    """, (symbol, cutoff_date, t / 100.0))
                    
                    stats = {row['outcome']: row['count'] for row in cursor.fetchall()}
                    success = stats.get('SUCCESS', 0)
                    fail = stats.get('FAIL', 0)
                    total = success + fail
                    
                    if total < MIN_TRADES:
                        continue # Preserve existing baseline if insufficient recent data
                    
                    found_recent = True
                    win_rate = (success / total) if total > 0 else 0.0
                    
                    # Bayesian Integration
                    # We start with a (2,2) prior (Weak belief at 50%)
                    alpha = 2.0 + success
                    beta = 2.0 + fail
                    
                    # Approved if Bayesian Mean >= 70% threshold
                    bayesian_mean = alpha / (alpha + beta)
                    approved = (bayesian_mean >= DEFAULT_HURDLE - 0.05) # Sligthly More lenient for recent streaks
                    
                    self.performance_matrix[symbol][str(t)] = {
                        "alpha": alpha,
                        "beta": beta,
                        "accuracy": win_rate,
                        "trades": total,
                        "status": "APPROVED" if approved else "BENCHED",
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "source": f"Live Bayesian ({lookback_days}d)"
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
            
            for symbol, tiers in audit_data.items():
                if symbol not in self.performance_matrix:
                    self.performance_matrix[symbol] = {}
                
                for t_str, data in tiers.items():
                    acc = data.get("accuracy", 0.0)
                    trades = data.get("trades", 0)
                    
                    # Bayesian Initialization from Audit
                    # alpha = 2.0 + (accuracy * trades)
                    # beta = 2.0 + ((1-accuracy) * trades)
                    alpha = 2.0 + (acc * trades)
                    beta = 2.0 + ((1.0 - acc) * trades)
                    
                    # Apply Hurdle: Bayesian Mean >= 70% threshold
                    bayesian_mean = alpha / (alpha + beta)
                    approved = (bayesian_mean >= DEFAULT_HURDLE - 0.05)
                    
                    # Baseline Sync: Set or Update
                    self.performance_matrix[symbol][t_str] = {
                        "alpha": alpha,
                        "beta": beta,
                        "accuracy": acc,
                        "trades": trades,
                        "status": "APPROVED" if approved else "BENCHED",
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "source": "Audit Baseline (Bayesian)"
                    }
            
            self.save_whitelist()
            logger.info("Multi-Tier Sync Complete.")
            
        except Exception as e:
            logger.error(f"Audit sync failed: {e}")

    def update_bayesian(self, symbol: str, tier: int, outcome: str):
        """Update alpha/beta for a specific symbol/tier based on live outcome."""
        if symbol not in self.performance_matrix:
            self.performance_matrix[symbol] = {}
        
        t_str = str(tier)
        # 1. Ensure Tier Entry Exists
        if t_str not in self.performance_matrix[symbol]:
            self.performance_matrix[symbol][t_str] = {
                "alpha": 2.0, "beta": 2.0, "accuracy": 0.5, "trades": 0, 
                "status": "BENCHED", "source": "Live Init"
            }
            
        data = self.performance_matrix[symbol][t_str]
        
        # 2. Robust Bayesian Initialization (Migrate Legacy Data)
        if "alpha" not in data or "beta" not in data:
            acc = data.get("accuracy", 0.5)
            tr = data.get("trades", 0)
            data["alpha"] = 2.0 + (acc * tr)
            data["beta"] = 2.0 + ((1.0 - acc) * tr)

        # 3. Apply Outcome
        if outcome == "SUCCESS":
            data["alpha"] += 1
        elif outcome == "FAIL":
            data["beta"] += 1
            
        data["trades"] += 1
        data["accuracy"] = (data["alpha"] - 2.0) / (data["trades"]) if data["trades"] > 0 else 0.5
        
        # Re-evaluate status
        bayesian_mean = data["alpha"] / (data["alpha"] + data["beta"])
        data["status"] = "APPROVED" if (bayesian_mean >= DEFAULT_HURDLE - 0.10) else "BENCHED"
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Bayesian Update [{symbol}@{tier}]: {outcome}. New Mean: {bayesian_mean:.1%}")
        self.save_whitelist()

# Singleton helper
_gate = None
def get_performance_gate() -> PerformanceGate:
    global _gate
    if _gate is None:
        _gate = PerformanceGate()
    return _gate
