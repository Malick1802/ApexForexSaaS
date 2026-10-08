# =============================================================================
# ApexForex SaaS - Dynamic Model Whitelist Engine
# =============================================================================
"""
Central authority governing dynamic, model-specific YTD winning asset whitelists.
Every strategy model in the platform (Confluence ML P60, Confluence Std P25, 
Confluence ML M15, Confluence Std M15, Foundation V1 Macro AI) has its own distinct edge profile.

Cycle & Rules:
1. Every day at 00:00 UTC rollover (or on daily scheduler tick / startup), the engine:
   - Queries signals.db for all closed trades Year-To-Date (YTD).
   - Computes Realized Net R for every pair independently per model using robust trade evaluation.
   - Designates winning assets (pairs with Net R >= 0.0, at least breakeven).
   - Persists the active matrix to config/dynamic_model_whitelists.json.
2. In the live trade execution loop (both Confluence scanner and Foundation executive):
   - Only pairs on that model's current active winning asset whitelist are permitted to submit broker orders.
   - Chronic underperforming assets (Net R < 0.0) are automatically demoted to background SHADOW trades,
     protecting live capital while preserving telemetry for continuous re-evaluation.
3. The cycle repeats autonomously every day.
"""

import os
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import yaml
import re
import pandas as pd
import numpy as np

PROFIT_REGEX = re.compile(r'(?:Profit:\s*)?\$([+-]?[\d,.]+)')

logger = logging.getLogger("DynamicModelWhitelist")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
WHITELIST_JSON = PROJECT_ROOT / "config" / "dynamic_model_whitelists.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "signals.db"

_manager_instance = None
_manager_lock = threading.Lock()


def normalize_symbol(symbol: str) -> str:
    """Normalize commodity/currency symbols across MT5 representations."""
    if not symbol:
        return ""
    s = str(symbol).strip().upper()
    if s == "GOLD":
        return "XAUUSD"
    if s == "SILVER":
        return "XAGUSD"
    if s in ("USOIL", "WTI", "CRUDEOIL"):
        return "USOIL.cash"
    if s in ("UKOIL", "BRENT"):
        return "UKOIL.cash"
    return s


def normalize_model_key(model_version: Optional[str]) -> str:
    """Map arbitrary model version tags to the canonical model key."""
    if not model_version:
        return "foundation_v1"
    m = str(model_version).strip().lower()
    if m in ("confluence_ml_p60", "confluence_ml_m15_p60", "apex-ml-p60", "ml_p60"):
        return "confluence_ml_p60"
    elif m in ("confluence_std_p25", "confluence_m15_p25", "apex-std-p25", "std_p25"):
        return "confluence_std_p25"
    elif m in ("confluence_ml_m15", "confluence_ml", "apex-ml"):
        return "confluence_ml_m15"
    elif m in ("confluence_m15", "confluence_standard", "confluence", "apex-std"):
        return "confluence_m15"
    elif m in ("v1", "foundation_v1", "foundation_tft", "foundation"):
        return "foundation_v1"
    elif m in ("manual_m15", "manual", "apex-manual"):
        return "manual_m15"
    return m


DEFAULT_SUB_MODELS = {
    "confluence_ml_p60": True,
    "confluence_std_p25": True,
    "confluence_ml_m15": True,
    "confluence_m15": True,
    "foundation_v1": True,
    "manual_m15": False,
}

SUB_MODEL_METADATA = {
    "confluence_ml_p60": {
        "key": "confluence_ml_p60",
        "name": "🧠 Confluence ML M15 (60% TP + 2p BE)",
        "short_name": "ML M15 (60% TP + 2p BE)",
        "badge": "🧠 ML P60",
        "description": "Wick-Aware ML model with LightGBM AI Quality Gate, 60% partial take-profit, and +2p spread breakeven.",
    },
    "confluence_std_p25": {
        "key": "confluence_std_p25",
        "name": "⚡ Confluence Standard M15 (25% TP + 2p BE)",
        "short_name": "Standard M15 (25% TP + 2p BE)",
        "badge": "⚡ Std P25",
        "description": "Standard Wick-Aware model without ML suppression, 25% partial take-profit, and +2p spread breakeven.",
    },
    "confluence_ml_m15": {
        "key": "confluence_ml_m15",
        "name": "🧠 Confluence ML M15 (AI Gate Fixed 1.5R)",
        "short_name": "Confluence AI Gate (1.5R)",
        "badge": "🧠 ML 1.5R",
        "description": "3-candle liquidity sweep with LightGBM win probability gate and fixed 1:1.5 RRR.",
    },
    "confluence_m15": {
        "key": "confluence_m15",
        "name": "⚡ Confluence Standard M15 (Rule-Based Fixed 1.5R)",
        "short_name": "Confluence Standard (1.5R)",
        "badge": "⚡ Std 1.5R",
        "description": "Pure 3-candle Day/Swing breakout without ML filtering and fixed 1:1.5 RRR.",
    },
    "foundation_v1": {
        "key": "foundation_v1",
        "name": "🌐 Foundation V1 Macro AI",
        "short_name": "Foundation Macro AI",
        "badge": "🌐 Foundation V1",
        "description": "Hourly macro neural network evaluating global yields, GMM regime, and 31-pair sequences.",
    },
    "manual_m15": {
        "key": "manual_m15",
        "name": "🎯 Manual M15 Wick Sniper",
        "short_name": "M15 Wick Sniper",
        "badge": "🎯 Manual Sniper",
        "description": "Discretionary terminal trades arming candle wick breakout entries.",
    },
}


class DynamicModelWhitelistManager:
    """
    Manages daily dynamic YTD winning asset whitelists per strategy model.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, json_path: Path = WHITELIST_JSON):
        self.db_path = Path(db_path)
        self.json_path = Path(json_path)
        self._cache: Dict[str, Any] = {}
        self._approved_set: Set[Tuple[str, str]] = set()
        self._last_mtime: float = 0.0
        self._last_refresh_date_utc: str = ""
        self._last_config_check: float = 0.0
        self._enabled_cached: bool = True
        self._last_disk_check: float = 0.0
        self._sub_models_cache: Dict[str, bool] = {}
        self._last_sub_models_check: float = 0.0
        self.load_from_disk(force=True)

    def _is_enabled_in_config(self) -> bool:
        """Check if dynamic YTD model whitelist filtering is enabled in config.yaml (throttled)."""
        now = time.time()
        if now - self._last_config_check < 2.0:
            return self._enabled_cached
        self._last_config_check = now
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                self._enabled_cached = bool(cfg.get("dynamic_model_whitelist", {}).get("enabled", True))
                return self._enabled_cached
        except Exception as e:
            logger.warning(f"Error checking dynamic_model_whitelist config: {e}")
        return True

    def get_sub_models_config(self) -> Dict[str, bool]:
        """Return dict of model_key -> bool indicating if model is activated under Dynamic YTD model (cached)."""
        now = time.time()
        if self._sub_models_cache and (now - self._last_sub_models_check < 5.0):
            return dict(self._sub_models_cache)
        self._last_sub_models_check = now

        cfg_dict = dict(DEFAULT_SUB_MODELS)
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                dyn_cfg = cfg.get("dynamic_model_whitelist", {})
                models_cfg = dyn_cfg.get("models", {})
                if isinstance(models_cfg, dict):
                    for k, v in models_cfg.items():
                        norm_k = normalize_model_key(k)
                        cfg_dict[norm_k] = bool(v)
        except Exception as e:
            logger.warning(f"Error reading sub-models from config: {e}")
        self._sub_models_cache = dict(cfg_dict)
        return cfg_dict

    def get_approved_set(self) -> Set[Tuple[str, str]]:
        """Return the current set of (normalized_model, normalized_symbol) tuples approved for Dynamic YTD."""
        self.load_from_disk()
        return set(self._approved_set)

    def is_model_active_under_ytd(self, model_version: Optional[str]) -> bool:
        """Check if a specific model is activated under the Dynamic YTD strategy."""
        if not model_version:
            return False
        m_canonical = normalize_model_key(model_version)
        cfg_dict = self.get_sub_models_config()
        return bool(cfg_dict.get(m_canonical, False))

    def set_sub_model_status(self, model_key: str, is_active: bool) -> bool:
        """Activate or deactivate a single model under the Dynamic YTD strategy."""
        cfg_dict = self.get_sub_models_config()
        norm_k = normalize_model_key(model_key)
        cfg_dict[norm_k] = bool(is_active)
        return self.save_sub_models_config(cfg_dict)

    def save_sub_models_config(self, sub_models: Dict[str, bool]) -> bool:
        """Save active sub-models under dynamic_model_whitelist in config.yaml."""
        try:
            cfg = {}
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            if "dynamic_model_whitelist" not in cfg or not isinstance(cfg["dynamic_model_whitelist"], dict):
                cfg["dynamic_model_whitelist"] = {}
            if "models" not in cfg["dynamic_model_whitelist"] or not isinstance(cfg["dynamic_model_whitelist"]["models"], dict):
                cfg["dynamic_model_whitelist"]["models"] = {}
            
            for k, v in sub_models.items():
                norm_k = normalize_model_key(k)
                cfg["dynamic_model_whitelist"]["models"][norm_k] = bool(v)

            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            # Invalidate caches and reload
            self._sub_models_cache = dict(cfg["dynamic_model_whitelist"]["models"])
            self._last_sub_models_check = time.time()
            self._approved_set = set()
            self._last_disk_check = 0.0
            self.load_from_disk(force=True)
            logger.info(f"Updated dynamic YTD model sub-models configuration: {sub_models}")
            return True
        except Exception as e:
            logger.error(f"Error saving dynamic model whitelist sub-models: {e}")
            return False

    def load_from_disk(self, force: bool = False) -> bool:
        """Load whitelist JSON from disk if modified (throttled)."""
        now = time.time()
        if not force and (now - self._last_disk_check < 2.0):
            return True
        self._last_disk_check = now
        try:
            if self.json_path.exists():
                mtime = os.path.getmtime(self.json_path)
                if mtime != self._last_mtime or force:
                    with open(self.json_path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                    self._last_mtime = mtime
                    self._last_refresh_date_utc = self._cache.get("as_of_date", "")

                    app_set = set()
                    active_sub = self.get_sub_models_config()
                    for m_k, m_v in self._cache.get("models", {}).items():
                        norm_m = normalize_model_key(m_k)
                        # Only include approved winning pairs for sub-models that are ACTIVATED under Dynamic YTD
                        if not active_sub.get(norm_m, False):
                            continue
                        for p in m_v.get("winning_pairs", []):
                            app_set.add((norm_m, normalize_symbol(p)))
                    self._approved_set = app_set
                    logger.info(f"Loaded dynamic model whitelists from disk (As of: {self._last_refresh_date_utc}, {len(self._approved_set)} rules across active models)")
                return True
            else:
                logger.info(f"Dynamic whitelist file not found at {self.json_path}. Recomputing now...")
                self.compute_ytd_whitelists()
                return True
        except Exception as e:
            logger.error(f"Failed to load dynamic model whitelists: {e}")
            return False

    def save_to_disk(self) -> bool:
        """Save active whitelist matrix to disk atomically."""
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.json_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            os.replace(temp_path, self.json_path)
            self._last_mtime = os.path.getmtime(self.json_path)
            logger.info(f"Persisted dynamic model whitelists to {self.json_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dynamic model whitelists: {e}")
            return False

    @staticmethod
    def _calc_trade_r(r: pd.Series, p_rule: Optional[str] = None, reward_mult: float = 1.5, risk: float = 50.0) -> pd.Series:
        """
        Calculate realized R, PnL, and status for a closed trade using institutional exit auditing.
        """
        reason = str(r.get('exit_reason') or '')
        outcome = r.get('outcome')
        model_ver = str(r.get('model_version') or '')
        if p_rule:
            model_ver = p_rule

        m = PROFIT_REGEX.search(reason)
        raw_profit = float(m.group(1).replace(',', '')) if m else None

        entry = r.get('price_at_signal')
        sl = r.get('sl_price')
        exit_p = r.get('exit_price')
        sig = r.get('signal')
        price_r = None
        if entry and sl and exit_p and entry != sl and not pd.isna(entry) and not pd.isna(sl) and not pd.isna(exit_p):
            risk_dist = abs(entry - sl)
            gain_dist = (exit_p - entry) if sig == 'BUY' else (entry - exit_p)
            price_r = gain_dist / risk_dist

        is_be = False
        if raw_profit is not None and abs(raw_profit) < 2.0:
            is_be = True
        elif price_r is not None and abs(price_r) < 0.12 and outcome == 'FAIL':
            is_be = True
        elif "BE hit" in reason or "Breakeven" in reason or "SL hit ($0.00)" in reason:
            is_be = True

        if is_be:
            return pd.Series([0.0, 0.0, 'BREAKEVEN'], index=['realized_r', 'pnl_amount', 'trade_status'])

        if outcome == 'FAIL':
            r_val = -1.0
            if price_r is not None and -1.2 <= price_r <= -0.5:
                r_val = round(price_r, 2)
            return pd.Series([r_val, r_val * risk, 'LOSS'], index=['realized_r', 'pnl_amount', 'trade_status'])

        if outcome == 'SUCCESS':
            if "p60" in model_ver:
                if raw_profit is not None and 0 < raw_profit < 35.0:
                    r_val = 0.90
                elif price_r is not None and price_r >= 1.4:
                    r_val = 1.50
                else:
                    r_val = 1.15
            elif "p25" in model_ver:
                if raw_profit is not None and 0 < raw_profit < 25.0:
                    r_val = 0.38
                elif price_r is not None and price_r >= 1.4:
                    r_val = 1.50
                else:
                    r_val = 0.94
            else:
                if "Friday" in reason and price_r is not None and price_r > 0:
                    r_val = round(min(reward_mult, max(0.2, price_r)), 2)
                else:
                    r_val = reward_mult
            return pd.Series([r_val, r_val * risk, 'WIN'], index=['realized_r', 'pnl_amount', 'trade_status'])

        return pd.Series([0.0, 0.0, 'OTHER'], index=['realized_r', 'pnl_amount', 'trade_status'])

    def compute_ytd_whitelists(self, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Recomputes YTD winning assets for every model based on all closed trades.
        as_of_date: Format 'YYYY-MM-DD'. If None, uses current UTC day start (midnight).
        """
        logger.info(f"🔄 Recomputing Dynamic Model Whitelists YTD (As of: {as_of_date or 'Today UTC Rollover'})...")
        from core.performance_report import PerformanceReporter

        reporter = PerformanceReporter(db_path=str(self.db_path))
        df_all = reporter._get_signals_df()
        if df_all.empty:
            logger.warning("No signals found in database. Using empty whitelist matrix.")
            return {}

        df_all['t_exit_utc'] = pd.to_datetime(df_all['exit_time'], format='ISO8601', utc=True)
        df_all['time_metric'] = df_all['t_exit_utc'].fillna(df_all['t_utc'])
        df_closed = df_all[df_all['outcome'].isin(['SUCCESS', 'FAIL'])].copy()

        now_utc = datetime.now(timezone.utc)
        if as_of_date:
            cutoff_dt = pd.to_datetime(f"{as_of_date} 00:00:00", utc=True)
            curr_date_str = as_of_date
        else:
            cutoff_dt = pd.to_datetime(now_utc.strftime("%Y-%m-%d 00:00:00"), utc=True)
            curr_date_str = now_utc.strftime("%Y-%m-%d")

        # Start of current year
        ytd_start_dt = pd.to_datetime(f"{cutoff_dt.year}-01-01 00:00:00", utc=True)

        # Filter strictly to YTD closed trades up to the cutoff
        df_ytd = df_closed[(df_closed['time_metric'] >= ytd_start_dt) & (df_closed['time_metric'] < cutoff_dt)].copy()
        
        # If today is the very start of the year or no trades before cutoff, fallback to include all closed trades
        if df_ytd.empty:
            df_ytd = df_closed[df_closed['time_metric'] >= ytd_start_dt].copy()

        # Model Definitions & Extraction Rules
        model_defs = {
            "confluence_ml_p60": {
                "name": "🧠 Confluence ML M15 (60% TP + 2p BE)",
                "fn": lambda df: df[df['model_version'].isin(['confluence_ml_p60', 'confluence_ml_m15_p60']) | ((df['model_version'] == 'confluence_ml_m15') & (df['exit_reason'] != 'ML_SUPPRESSED'))],
                "p_rule": "p60",
                "min_r_threshold": 0.0,
            },
            "confluence_std_p25": {
                "name": "⚡ Confluence Standard M15 (25% TP + 2p BE)",
                "fn": lambda df: df[df['model_version'].isin(['confluence_std_p25', 'confluence_m15_p25']) | (df['model_version'] == 'confluence_m15')],
                "p_rule": "p25",
                "min_r_threshold": 0.0,
            },
            "confluence_ml_m15": {
                "name": "🧠 Confluence ML M15 (AI Gate Fixed 1.5R)",
                "fn": lambda df: df[(df['model_version'] == 'confluence_ml_m15') & (df['exit_reason'] != 'ML_SUPPRESSED')],
                "p_rule": None,
                "min_r_threshold": 0.0,
            },
            "confluence_m15": {
                "name": "⚡ Confluence Standard M15 (Rule-Based Fixed 1.5R)",
                "fn": lambda df: df[df['model_version'] == 'confluence_m15'],
                "p_rule": None,
                "min_r_threshold": 0.0,
            },
            "foundation_v1": {
                "name": "🌐 Foundation V1 Macro AI",
                "fn": lambda df: df[df['model_version'].isin(['v1', 'foundation_tft', 'foundation', 'foundation_v1']) | df['model_version'].isnull()],
                "p_rule": None,
                "min_r_threshold": 0.0,
            },
            "manual_m15": {
                "name": "🎯 Manual M15 Wick Sniper",
                "fn": lambda df: df[(df['is_manual'] == 1) | (df['model_version'] == 'manual_m15')],
                "p_rule": None,
                "min_r_threshold": 0.0,
            },
        }

        models_data = {}
        for m_key, m_info in model_defs.items():
            sub = m_info["fn"](df_ytd).copy()
            if sub.empty:
                models_data[m_key] = {
                    "name": m_info["name"],
                    "total_trades_ytd": 0,
                    "winning_pairs": [],
                    "benched_pairs": [],
                    "pair_stats": {},
                    "active_under_ytd": self.is_model_active_under_ytd(m_key),
                }
                continue

            sub_dedup = reporter._dedup(sub)
            c = sub_dedup.apply(lambda r: self._calc_trade_r(r, p_rule=m_info["p_rule"]), axis=1)
            sub_dedup['realized_r'] = c['realized_r']
            sub_dedup['pnl_amount'] = c['pnl_amount']
            sub_dedup['trade_status'] = c['trade_status']

            sym_stats = {}
            winning_pairs = []
            benched_pairs = []

            for sym, grp in sub_dedup.groupby('symbol'):
                sym_clean = normalize_symbol(sym)
                n_trades = len(grp)
                n_wins = int((grp['trade_status'] == 'WIN').sum())
                n_losses = int((grp['trade_status'] == 'LOSS').sum())
                n_be = int((grp['trade_status'] == 'BREAKEVEN').sum())
                net_r = round(float(grp['realized_r'].sum()), 2)
                pnl_usd = round(float(grp['pnl_amount'].sum()), 2)
                wr = round(float(n_wins / n_trades * 100), 1) if n_trades > 0 else 0.0

                is_winning = (net_r >= m_info["min_r_threshold"])
                if is_winning:
                    winning_pairs.append(sym_clean)
                else:
                    benched_pairs.append(sym_clean)

                sym_stats[sym_clean] = {
                    "trades": n_trades,
                    "wins": n_wins,
                    "losses": n_losses,
                    "be": n_be,
                    "net_r": net_r,
                    "pnl_usd": pnl_usd,
                    "win_rate": wr,
                    "status": "APPROVED" if is_winning else "BENCHED",
                }

            winning_pairs = sorted(list(set(winning_pairs)))
            benched_pairs = sorted(list(set(benched_pairs)))

            models_data[m_key] = {
                "name": m_info["name"],
                "total_trades_ytd": len(sub_dedup),
                "winning_pairs": winning_pairs,
                "benched_pairs": benched_pairs,
                "pair_stats": sym_stats,
                "active_under_ytd": self.is_model_active_under_ytd(m_key),
            }

        payload = {
            "last_updated_utc": now_utc.isoformat(),
            "as_of_date": curr_date_str,
            "min_r_hurdle": 0.0,
            "sub_models_config": self.get_sub_models_config(),
            "models": models_data,
        }

        self._cache = payload
        self._last_refresh_date_utc = curr_date_str
        self.save_to_disk()

        for k, v in models_data.items():
            status_tag = "ACTIVE" if v.get('active_under_ytd', True) else "DISABLED"
            logger.info(
                f"✅ Dynamic Whitelist [{k}] ({status_tag}): {len(v['winning_pairs'])} winning pairs approved "
                f"({len(v['benched_pairs'])} benched | {v['total_trades_ytd']} YTD trades)"
            )

        return self._cache

    def check_daily_refresh(self) -> bool:
        """
        Verifies if the current UTC calendar date is newer than the last computation.
        If a new calendar day has begun, automatically triggers compute_ytd_whitelists().
        """
        now_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_refresh_date_utc != now_date_str:
            logger.info(
                f"🌅 UTC Day Rollover detected ({self._last_refresh_date_utc} -> {now_date_str}). "
                f"Triggering autonomous daily Dynamic Model Whitelist recalculation..."
            )
            self.compute_ytd_whitelists()
            return True
        return False

    def is_pair_approved(self, model_version: Optional[str], symbol: str) -> bool:
        """
        Main Execution Gate:
        Checks whether 'symbol' is an authorized winning asset (Net R >= 0.0) for 'model_version'.
        If the dynamic whitelist engine is disabled in config.yaml, returns True (pass-through).
        If the model itself is DEACTIVATED under the Dynamic YTD Model, returns False.
        """
        if not self._is_enabled_in_config():
            return True

        m_canonical = normalize_model_key(model_version)
        if not self.is_model_active_under_ytd(m_canonical):
            return False

        self.load_from_disk()

        sym_clean = normalize_symbol(symbol)
        if (m_canonical, sym_clean) in self._approved_set:
            return True

        # If model not in cache at all, allow as fallback only if active
        if m_canonical not in self._cache.get("models", {}):
            return True

        return False

    def get_winning_pairs(self, model_version: str) -> List[str]:
        """Return list of approved winning pairs for a model."""
        self.load_from_disk()
        m_canonical = normalize_model_key(model_version)
        model_entry = self._cache.get("models", {}).get(m_canonical, {})
        return list(model_entry.get("winning_pairs", []))

    def get_benched_pairs(self, model_version: str) -> List[str]:
        """Return list of benched underperforming pairs for a model."""
        self.load_from_disk()
        m_canonical = normalize_model_key(model_version)
        model_entry = self._cache.get("models", {}).get(m_canonical, {})
        return list(model_entry.get("benched_pairs", []))

    def get_model_summary(self, model_version: str) -> Dict[str, Any]:
        """Return full summary and per-pair metrics for a model."""
        self.load_from_disk()
        m_canonical = normalize_model_key(model_version)
        return self._cache.get("models", {}).get(m_canonical, {})

    def get_all_models_summary(self) -> Dict[str, Any]:
        """Return all models matrix."""
        self.load_from_disk()
        return self._cache


def get_dynamic_whitelist_manager() -> DynamicModelWhitelistManager:
    """Singleton factory for DynamicModelWhitelistManager."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = DynamicModelWhitelistManager()
        return _manager_instance


def is_pair_whitelisted_for_model(model_version: Optional[str], symbol: str) -> bool:
    """Convenience helper to check if symbol is approved for model."""
    return get_dynamic_whitelist_manager().is_pair_approved(model_version, symbol)


def is_model_active_under_ytd(model_version: Optional[str]) -> bool:
    """Convenience helper to check if a model is activated under Dynamic YTD model."""
    return get_dynamic_whitelist_manager().is_model_active_under_ytd(model_version)


def set_ytd_sub_model_status(model_key: str, is_active: bool) -> bool:
    """Convenience helper to activate or deactivate a sub-model under Dynamic YTD model."""
    return get_dynamic_whitelist_manager().set_sub_model_status(model_key, is_active)


def get_ytd_sub_models_config() -> Dict[str, bool]:
    """Convenience helper to get all sub-models activation statuses under Dynamic YTD model."""
    return get_dynamic_whitelist_manager().get_sub_models_config()


def get_ytd_model_attribution(model_version: Optional[str], symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Determine if a trade/signal is governed by or generated under the Dynamic YTD Model.
    Provides formatted strings and badges for Telegram alerts, logs, and dashboards.
    """
    manager = get_dynamic_whitelist_manager()
    m_canonical = normalize_model_key(model_version)
    sym_clean = normalize_symbol(symbol) if symbol else ""

    ytd_enabled = manager._is_enabled_in_config()
    sub_active = manager.is_model_active_under_ytd(m_canonical) if ytd_enabled else False
    is_winning = manager.is_pair_approved(m_canonical, sym_clean) if (ytd_enabled and sym_clean) else False

    # Is this signal generated under the Dynamic YTD Model architecture?
    is_ytd = (m_canonical == "dynamic_ytd_model") or (ytd_enabled and sub_active and (is_winning or not sym_clean))

    meta = SUB_MODEL_METADATA.get(m_canonical, {})
    sub_name = meta.get("short_name", m_canonical)
    badge = meta.get("badge", m_canonical)

    if is_ytd:
        display_model_name = f"Dynamic YTD Model ({sub_name})"
        tag = "🏆 [Dynamic YTD Model]"
    else:
        display_model_name = sub_name if sub_name != m_canonical else m_canonical
        tag = ""

    return {
        "is_ytd": is_ytd,
        "ytd_enabled": ytd_enabled,
        "is_sub_model_active": sub_active,
        "is_winning_asset": is_winning,
        "canonical_model": m_canonical,
        "sub_model_name": sub_name,
        "sub_model_badge": badge,
        "display_model_name": display_model_name,
        "tag": tag,
    }
