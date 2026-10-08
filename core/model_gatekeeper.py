# =============================================================================
# ApexForex Unified Model Gatekeeper
# =============================================================================
"""
Central authority governing which strategy models are authorized to submit
LIVE orders to MT5 vs running in background SHADOW (Paper Trading) mode.

Rules:
1. ONLY models explicitly enabled for LIVE execution can submit broker orders.
2. ALL models not enabled for LIVE execution run in background SHADOW mode.
3. Every model's performance (Live & Shadow) is tracked in signals.db and the
   Performance Matrix for comprehensive side-by-side comparison.
"""

from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
import logging

logger = logging.getLogger("ModelGatekeeper")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Supported Strategy Engines
SUPPORTED_MODELS = {
    "confluence_ml_p60": {
        "key": "confluence_ml_p60",
        "name": "🧠 Confluence ML M15 (60% TP Partial + 2p Spread BE)",
        "short_name": "ML M15 (60% TP + 2p BE)",
        "db_versions": ["confluence_ml_p60", "confluence_ml_m15_p60", "APEX-ML-P60"],
        "default_live": True,
        "description": "Wick-Aware ML model with LightGBM AI Quality Gate. Takes partial profit at 60% of TP and moves SL to +2.0 pips beyond entry to compensate for spread.",
        "partial_ratio": 0.60,
        "be_offset_pips": 2.0,
        "allow_concurrent_asset": True,
    },
    "confluence_std_p25": {
        "key": "confluence_std_p25",
        "name": "⚡ Confluence Standard M15 (25% TP Partial + 2p Spread BE)",
        "short_name": "Standard M15 (25% TP + 2p BE)",
        "db_versions": ["confluence_std_p25", "confluence_m15_p25", "APEX-STD-P25"],
        "default_live": True,
        "description": "Standard Wick-Aware model without ML suppression. Takes partial profit at 25% of TP and moves SL to +2.0 pips beyond entry to compensate for spread.",
        "partial_ratio": 0.25,
        "be_offset_pips": 2.0,
        "allow_concurrent_asset": True,
    },
    "confluence_ml_m15": {
        "key": "confluence_ml_m15",
        "name": "🧠 Confluence M15 + Deep Learning (AI Gate - Fixed 1.5R)",
        "short_name": "Confluence AI Gate",
        "db_versions": ["confluence_ml_m15"],
        "default_live": False,
        "description": "3-candle liquidity sweep with LightGBM 44-feature win probability gate (P(win) >= 48%).",
        "partial_ratio": None,
        "be_offset_pips": 0.0,
        "allow_concurrent_asset": False,
    },
    "confluence_m15": {
        "key": "confluence_m15",
        "name": "⚡ Confluence M15 Standard (Rule-Based - Fixed 1.5R)",
        "short_name": "Confluence Standard",
        "db_versions": ["confluence_m15"],
        "default_live": False,
        "description": "Pure 3-candle Day/Swing line breakout and reclamation without ML filtering.",
        "partial_ratio": None,
        "be_offset_pips": 0.0,
        "allow_concurrent_asset": False,
    },
    "foundation_v1": {
        "key": "foundation_v1",
        "name": "🌐 Foundation V1 Macro AI",
        "short_name": "Foundation Macro AI",
        "db_versions": ["v1", "foundation_tft", "foundation"],
        "default_live": False,
        "description": "Hourly macro neural network evaluating global yields, GMM regime, and 31-pair sequences (61%+ floor).",
    },
    "manual_m15": {
        "key": "manual_m15",
        "name": "🎯 Manual M15 Wick Sniper (Discretionary)",
        "short_name": "M15 Wick Sniper",
        "db_versions": ["manual_m15", "MANUAL"],
        "default_live": False,
        "description": "Discretionary terminal button that arms candle wick breakout entries.",
    },
    "dynamic_ytd_model": {
        "key": "dynamic_ytd_model",
        "name": "🌟 Dynamic YTD Model (Daily Winning Asset Strategy)",
        "short_name": "Dynamic YTD Model",
        "db_versions": ["dynamic_ytd_model", "dynamic_ytd", "dynamic_model", "DYNAMIC_YTD"],
        "default_live": True,
        "description": "Autonomous multi-model strategy that dynamically updates YTD profitable pairs (Net R >= 0.0) every day for all activated models, trading winning assets on live MT5 while repeating the cycle autonomously every 24h.",
        "partial_ratio": None,
        "be_offset_pips": 0.0,
        "allow_concurrent_asset": True,
    },
}


def load_gatekeeper_config() -> Dict[str, bool]:
    """Load live authorization status for each model from config.yaml."""
    cfg = {m_key: meta["default_live"] for m_key, meta in SUPPORTED_MODELS.items()}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                if "model_gatekeeper" in raw and isinstance(raw["model_gatekeeper"], dict):
                    for k, v in raw["model_gatekeeper"].items():
                        if k in cfg:
                            cfg[k] = bool(v)
                # Keep synchronized with confluence_model keys if present
                conf_cfg = raw.get("confluence_model", {})
                if "enable_ml_p60_model" in conf_cfg:
                    cfg["confluence_ml_p60"] = bool(conf_cfg["enable_ml_p60_model"])
                if "enable_std_p25_model" in conf_cfg:
                    cfg["confluence_std_p25"] = bool(conf_cfg["enable_std_p25_model"])
                if "enable_ml_model" in conf_cfg:
                    cfg["confluence_ml_m15"] = bool(conf_cfg["enable_ml_model"])
                if "enable_standard_model" in conf_cfg:
                    cfg["confluence_m15"] = bool(conf_cfg["enable_standard_model"])
                
                # Keep synchronized with dynamic_model_whitelist section
                dyn_cfg = raw.get("dynamic_model_whitelist", {})
                if "enabled" in dyn_cfg:
                    cfg["dynamic_ytd_model"] = bool(dyn_cfg["enabled"])
        except Exception as e:
            logger.warning(f"Error reading model gatekeeper config: {e}")
    return cfg


def save_gatekeeper_config(model_statuses: Dict[str, bool]) -> bool:
    """Save live authorization status to config.yaml and synchronize related sections."""
    try:
        raw = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        gate_cfg = raw.get("model_gatekeeper", {})
        for k, v in model_statuses.items():
            gate_cfg[k] = bool(v)
        raw["model_gatekeeper"] = gate_cfg

        # Synchronize confluence_model section
        conf_cfg = raw.get("confluence_model", {})
        if "confluence_ml_p60" in model_statuses:
            conf_cfg["enable_ml_p60_model"] = bool(model_statuses["confluence_ml_p60"])
        if "confluence_std_p25" in model_statuses:
            conf_cfg["enable_std_p25_model"] = bool(model_statuses["confluence_std_p25"])
        if "confluence_ml_m15" in model_statuses:
            conf_cfg["enable_ml_model"] = bool(model_statuses["confluence_ml_m15"])
        if "confluence_m15" in model_statuses:
            conf_cfg["enable_standard_model"] = bool(model_statuses["confluence_m15"])
        raw["confluence_model"] = conf_cfg

        # Synchronize dynamic_model_whitelist section
        dyn_cfg = raw.get("dynamic_model_whitelist", {})
        if "dynamic_ytd_model" in model_statuses:
            dyn_cfg["enabled"] = bool(model_statuses["dynamic_ytd_model"])
        raw["dynamic_model_whitelist"] = dyn_cfg

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Error saving model gatekeeper config: {e}")
        return False


def is_model_live_authorized(model_version: Optional[str]) -> bool:
    """
    Check if a model version is authorized for LIVE MT5 order execution.
    If False, the model MUST only trade in background SHADOW (paper) mode.
    """
    if not model_version:
        return False
    
    cfg = load_gatekeeper_config()
    m_clean = str(model_version).strip().lower()

    # Map model_version strings to gatekeeper keys
    if m_clean in ("dynamic_ytd_model", "dynamic_ytd", "dynamic_model", "dynamic"):
        return bool(cfg.get("dynamic_ytd_model", True))
    elif m_clean in ("confluence_ml_p60", "confluence_ml_m15_p60", "apex-ml-p60", "ml_p60"):
        return bool(cfg.get("confluence_ml_p60", True))
    elif m_clean in ("confluence_std_p25", "confluence_m15_p25", "apex-std-p25", "std_p25"):
        return bool(cfg.get("confluence_std_p25", True))
    elif m_clean in ("confluence_ml_m15", "confluence_ml"):
        return bool(cfg.get("confluence_ml_m15", False))
    elif m_clean in ("confluence_m15", "confluence_standard", "confluence"):
        return bool(cfg.get("confluence_m15", False))
    elif m_clean in ("v1", "foundation_tft", "foundation_v1", "foundation"):
        return bool(cfg.get("foundation_v1", False))
    elif m_clean in ("manual_m15", "manual"):
        return bool(cfg.get("manual_m15", False))
    
    # Default to False (Safe: Never execute unknown models live)
    return False


def get_model_catalog() -> List[Dict[str, Any]]:
    """Return complete status catalog of all models with their live/shadow state."""
    cfg = load_gatekeeper_config()
    catalog = []
    for k, meta in SUPPORTED_MODELS.items():
        is_live = bool(cfg.get(k, meta["default_live"]))
        catalog.append({
            "key": k,
            "name": meta["name"],
            "short_name": meta["short_name"],
            "description": meta["description"],
            "is_live": is_live,
            "status": "LIVE" if is_live else "SHADOW",
            "badge": "🟢 LIVE ACTIVE" if is_live else "👻 SHADOW MODE",
        })
    return catalog


def get_live_models() -> List[str]:
    """Return list of model keys currently authorized for LIVE execution."""
    cfg = load_gatekeeper_config()
    return [k for k, v in cfg.items() if v]


def get_shadow_models() -> List[str]:
    """Return list of model keys currently restricted to SHADOW execution."""
    cfg = load_gatekeeper_config()
    return [k for k, v in cfg.items() if not v]


def get_model_catalog_with_whitelists() -> List[Dict[str, Any]]:
    """Return catalog of all models enriched with their dynamic YTD winning assets."""
    catalog = get_model_catalog()
    try:
        from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
        mgr = get_dynamic_whitelist_manager()
        summary = mgr.get_all_models_summary()
        models_data = summary.get("models", {})
        
        for item in catalog:
            m_key = item["key"]
            m_dyn = models_data.get(m_key, {})
            item["winning_pairs"] = m_dyn.get("winning_pairs", [])
            item["benched_pairs"] = m_dyn.get("benched_pairs", [])
            item["total_trades_ytd"] = m_dyn.get("total_trades_ytd", 0)
            item["pair_stats"] = m_dyn.get("pair_stats", {})
            item["as_of_date"] = summary.get("as_of_date", "")
            item["last_updated_utc"] = summary.get("last_updated_utc", "")
    except Exception as e:
        logger.warning(f"Error enriching catalog with dynamic whitelists: {e}")
    return catalog
