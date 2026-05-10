"""
MT5 History & Symbol Audit - Lightweight Version
=================================================
Uses copy_rates_from_pos to quickly check available bar count
without waiting for full server sync.
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.mt5_connector import get_mt5
    mt5 = get_mt5()
    if mt5 is None:
        print("[ERROR] MT5 bridge returned None. Is MetaTrader 5 open and logged in?")
        sys.exit(1)
    print("[OK] MT5 connected.\n")
except Exception as e:
    print(f"[ERROR] Could not connect to MT5: {e}")
    sys.exit(1)

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "GBPJPY", "EURJPY", "AUDJPY", "CADJPY", "CHFJPY",
    "NZDJPY", "GBPCHF", "EURGBP", "AUDNZD", "NZDCHF", "NZDCAD",
    "CADCHF", "AUDCHF", "EURCAD", "GBPNZD", "EURNZD", "GBPCAD",
    "USDSGD", "XAUUSD"
]

MACRO_CANDIDATES = {
    "S&P 500":   ["US500", "SP500", "USA500", "SPX500", "US500Cash"],
    "Crude Oil": ["USOIL", "WTI", "OIL", "XTIUSD", "USOILCash"],
    "NASDAQ":    ["NAS100", "USTEC", "US100", "NDX"],
    "Gold":      ["XAUUSD", "GOLD", "XAUUSDm"],
    "Silver":    ["XAGUSD", "SILVER"],
}

TF = mt5.TIMEFRAME_H1
# Request large but finite number of bars: 5 years * 24h * ~260 trading days = ~31200
MAX_BARS = 40000

print("=" * 65)
print("  FOREX PAIR HISTORY DEPTH (1H candles via copy_rates_from_pos)")
print("=" * 65)
print(f"{'Symbol':<12} | {'Oldest Bar':<15} | {'Bars':>8} | {'Years':>6}")
print("-" * 65)

pair_results = {}
for symbol in FOREX_PAIRS:
    try:
        mt5.symbol_select(symbol, True)
        # copy from position 0 going back MAX_BARS from the most recent bar
        rates = mt5.copy_rates_from_pos(symbol, TF, 0, MAX_BARS)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            start_dt = df['time'].iloc[0]
            end_dt = df['time'].iloc[-1]
            years = (end_dt - start_dt).days / 365.25
            pair_results[symbol] = years
            note = " *** CAPPED" if len(rates) == MAX_BARS else ""
            print(f"{symbol:<12} | {str(start_dt.date()):<15} | {len(df):>8,} | {years:>5.2f}y{note}")
        else:
            err = mt5.last_error()
            print(f"{symbol:<12} | NO DATA -- {err}")
    except Exception as e:
        print(f"{symbol:<12} | ERROR: {e}")

print()
print("=" * 65)
print("  MACRO ASSET SEARCH (checking known names on your broker)")
print("=" * 65)

all_symbols_obj = mt5.symbols_get()
all_names = [s.name for s in all_symbols_obj] if all_symbols_obj else []

macro_found = {}
for asset, candidates in MACRO_CANDIDATES.items():
    found = None
    for c in candidates:
        if c in all_names:
            found = c
            break
    if not found:
        for c in candidates:
            matches = [n for n in all_names if c.lower() in n.lower()]
            if matches:
                found = matches[0]
                break
    if found:
        mt5.symbol_select(found, True)
        rates = mt5.copy_rates_from_pos(found, TF, 0, MAX_BARS)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            years = (df['time'].iloc[-1] - df['time'].iloc[0]).days / 365.25
            note = " *** CAPPED" if len(rates) == MAX_BARS else ""
            macro_found[asset] = found
            print(f"  [FOUND] {asset:<15} -> '{found}' | {years:.2f}y | {len(df):,} bars{note}")
        else:
            print(f"  [FOUND] {asset:<15} -> '{found}' | No history")
    else:
        print(f"  [MISS]  {asset:<15} -> Not on broker (will use yfinance)")

print()
print("=" * 65)
print("  SUMMARY")
print("=" * 65)
if pair_results:
    min_years = min(pair_results.values())
    max_years = max(pair_results.values())
    bottleneck = [s for s,y in pair_results.items() if y == min_years]
    print(f"  Pairs with data       : {len(pair_results)}/{len(FOREX_PAIRS)}")
    print(f"  Shortest history      : {min_years:.2f} years ({bottleneck})")
    print(f"  Longest history       : {max_years:.2f} years")
    print(f"  Macro on MT5          : {len(macro_found)}/{len(MACRO_CANDIDATES)}")
    print(f"  Macro via yfinance    : {len(MACRO_CANDIDATES)-len(macro_found)}/{len(MACRO_CANDIDATES)}")
    print()
    if min_years >= 4.5:
        rec = "5-YEAR"
    elif min_years >= 2.8:
        rec = "3-YEAR"
    else:
        rec = f"{min_years:.1f}-YEAR (bottleneck limited)"
    print(f"  >>> RECOMMENDED TRAINING HORIZON: {rec}")
print("=" * 65)
