"""
core/symbol_guard.py
====================
Centralized Commodity and Blocked Symbol Security Guard.
Guarantees that commodities (Gold, Silver, Oil, Gas, Copper, Platinum, Palladium, Crypto, Indices)
are strictly blocked from executing live trades in MT5, multi-account copy executors, and terminal alerts.
"""

from typing import Set
import yaml
from pathlib import Path

# Known commodities and non-forex asset identifiers
COMMODITY_SYMBOLS: Set[str] = {
    # Gold
    "XAUUSD", "GOLD", "XAUUSD.CASH", "XAUUSD.M", "XAUUSD.RAW", "XAUUSD_", "XAUUSD.",
    # Silver
    "XAGUSD", "SILVER", "XAGUSD.CASH", "XAGUSD.M", "XAGUSD.RAW", "XAGUSD_", "XAGUSD.",
    # Oil & Energy
    "USOIL", "USOIL.CASH", "USOIL.M", "UKOIL", "UKOIL.CASH", "UKOIL.M",
    "BRENT", "WTI", "CRUDEOIL", "CL", "NGAS", "NATGAS", "GASOIL",
    # Metals
    "COPPER", "XPTUSD", "PLATINUM", "XPDUSD", "PALLADIUM", "ALUMINUM", "ZINC", "NICKEL",
    # Crypto / Indices (Safety fallback)
    "BTCUSD", "ETHUSD", "US30", "NAS100", "SPX500", "GER30", "GER40"
}

COMMODITY_PREFIXES = (
    "XAU", "XAG", "USOIL", "UKOIL", "GOLD", "SILVER", "BRENT", "WTI",
    "COPPER", "XPT", "XPD", "NGAS", "NATGAS", "BTC", "ETH"
)

def is_commodity(symbol: str) -> bool:
    """Return True if symbol is a commodity or non-forex asset."""
    if not symbol:
        return False
    clean = str(symbol).upper().strip()
    if clean in COMMODITY_SYMBOLS:
        return True
    for p in COMMODITY_PREFIXES:
        if clean.startswith(p):
            return True
    return False

# Central hardcoded blacklist fallback (defense-in-depth)
HARDCODED_BLOCKED_DIRECTIONS = {
    "EURUSD": {"BUY"},
    "EURCAD": {"BUY"},
    "XAUUSD": {"BUY"},
    "AUDUSD": {"BUY"}
}

import functools
import os
from typing import Tuple, Dict

@functools.lru_cache(maxsize=8)
def _get_cached_guard_config(cfg_path_str: str, mtime: float) -> Tuple[Set[str], Dict[str, Set[str]]]:
    try:
        with open(cfg_path_str, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
            blocked = set(str(s).upper().strip() for s in cfg.get('blocked_symbols', []))
            blocked_dirs = {}
            for sym, dirs in cfg.get('blocked_directions', {}).items():
                blocked_dirs[str(sym).upper().strip()] = set(str(d).upper().strip() for d in dirs)
            return blocked, blocked_dirs
    except Exception:
        return set(), {}

def get_guard_config(config_path: str = None) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Retrieve active blocked symbols and blocked directions with mtime caching."""
    cfg_file = Path(config_path) if config_path else (Path(__file__).resolve().parent.parent / "config.yaml")
    if not cfg_file.exists():
        cfg_file = Path("config.yaml")
    if cfg_file.exists():
        try:
            mtime = os.path.getmtime(str(cfg_file))
            return _get_cached_guard_config(str(cfg_file.resolve()), mtime)
        except Exception:
            pass
    return set(), {}

def is_symbol_blocked(symbol: str, config_path: str = None) -> bool:
    """
    Return True if symbol is explicitly listed in config.yaml `blocked_symbols`
    or identified as any commodity asset (permanent commodity ban).
    """
    if not symbol:
        return True
    
    clean = str(symbol).upper().strip()
    if is_commodity(clean):
        return True

    blocked, _ = get_guard_config(config_path)
    return clean in blocked

def is_direction_blocked(symbol: str, signal_type: str, config_path: str = None) -> bool:
    """
    Return True if the specific direction (e.g. EURUSD BUY) is blocked.
    """
    if not symbol or not signal_type:
        return True
    
    clean_sym = str(symbol).upper().strip()
    clean_sig = str(signal_type).upper().strip()

    if is_symbol_blocked(clean_sym, config_path):
        return True

    # 1. Hardcoded defense-in-depth check
    if clean_sym in HARDCODED_BLOCKED_DIRECTIONS and clean_sig in HARDCODED_BLOCKED_DIRECTIONS[clean_sym]:
        return True

    # 2. Config file check
    _, blocked_dirs = get_guard_config(config_path)
    if clean_sym in blocked_dirs and clean_sig in blocked_dirs[clean_sym]:
        return True

    return False


def is_commodity_benched(symbol: str, signal_type: str, config_path: str = None, db_path: str = None) -> bool:
    """
    Return True if symbol is a commodity (all commodities are banned from live execution).
    Forex pairs are never benched by this gate (always returns False for Forex).
    """
    if is_commodity(symbol):
        return True
        
    if not symbol or not signal_type:
        return False

    clean_sym = str(symbol).strip()
    clean_sig = str(signal_type).upper().strip()

    # If the direction is already blocked (e.g. XAUUSD BUY), treat as blocked
    if is_direction_blocked(clean_sym, clean_sig, config_path):
        return True

    # Check config settings for commodity benching gate
    cfg_file = Path(config_path) if config_path else (Path(__file__).resolve().parent.parent / "config.yaml")
    window = 5
    min_wr = 0.40
    enabled = True
    if cfg_file.exists():
        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                c_gate = cfg.get('safety', {}).get('commodity_40pct_gate', {})
                enabled = c_gate.get('enabled', True)
                min_wr = float(c_gate.get('min_win_rate', 0.40))
                window = int(c_gate.get('rolling_window', 5))
        except Exception:
            pass

    if not enabled:
        return False

    # Resolve DB path
    if not db_path:
        db_path = str(Path(__file__).resolve().parent.parent / "signals.db")

    if not Path(db_path).exists():
        return False

    try:
        import sqlite3
        with sqlite3.connect(db_path, timeout=5) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT outcome
                FROM signals
                WHERE symbol = ?
                  AND signal = ?
                  AND confidence >= 0.55
                  AND outcome IN ('SUCCESS', 'FAIL')
                ORDER BY timestamp DESC
                LIMIT ?
            """, (clean_sym, clean_sig, window))
            rows = cur.fetchall()
            if len(rows) < window:
                # Not enough trades to bench yet
                return False
            wins = sum(1 for r in rows if r[0] == 'SUCCESS')
            wr = wins / len(rows)
            return wr < min_wr
    except Exception:
        return False


def get_commodity_benching_stats(symbol: str, signal_type: str, config_path: str = None, db_path: str = None) -> dict:
    """Return detailed stats for commodity benching gate."""
    if not is_commodity(symbol):
        return {"is_commodity": False, "benched": False}

    clean_sym = str(symbol).strip()
    clean_sig = str(signal_type).upper().strip()
    window = 5
    min_wr = 0.40

    if not db_path:
        db_path = str(Path(__file__).resolve().parent.parent / "signals.db")

    if not Path(db_path).exists():
        return {"is_commodity": True, "benched": False, "trades": 0}

    try:
        import sqlite3
        with sqlite3.connect(db_path, timeout=5) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT outcome
                FROM signals
                WHERE symbol = ?
                  AND signal = ?
                  AND confidence >= 0.55
                  AND outcome IN ('SUCCESS', 'FAIL')
                ORDER BY timestamp DESC
                LIMIT ?
            """, (clean_sym, clean_sig, window))
            rows = cur.fetchall()
            tot = len(rows)
            wins = sum(1 for r in rows if r[0] == 'SUCCESS')
            wr = (wins / tot) if tot > 0 else 0.0
            benched = (tot >= window and wr < min_wr)
            return {
                "is_commodity": True,
                "symbol": clean_sym,
                "signal": clean_sig,
                "trades": tot,
                "wins": wins,
                "losses": tot - wins,
                "win_rate": wr,
                "benched": benched,
                "window": window,
                "threshold": min_wr
            }
    except Exception as e:
        return {"is_commodity": True, "benched": False, "error": str(e)}

