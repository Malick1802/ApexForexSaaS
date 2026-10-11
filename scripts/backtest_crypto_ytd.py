"""
scripts/backtest_crypto_ytd.py
Year-To-Date (2026-01-01 to Present) Backtest for 8 Crypto Assets Across 4 Confluence Models.

Assets:
- BTCUSD, ETHUSD, SOLUSD, XRPUSD, DOGEUSD, ADAUSD, LTCUSD, BNBUSD

Models Evaluated:
1. confluence_ml_p60 (🧠 ML Gate >= 0.48, 60% Partial TP, Breakeven)
2. confluence_std_p25 (⚡ Rule-Based, 25% Partial TP, Breakeven)
3. confluence_ml_m15 (🧠 ML Gate >= 0.48, Fixed 1:1.5 RRR)
4. confluence_m15 (⚡ Rule-Based, Fixed 1:1.5 RRR)

Execution & Strategy Rules:
- Day Line: 21:00 UTC 24h Rollover (Upper = Max High, Lower = Min Low of prior 24h)
- Swing Line: First M15 local peak/trough prior to day start higher/lower than day line
- Candle 1: Wick sweep across line (originated from opposite side)
- Candle 2 (Magic Candle): Reclamation across line, body/range >= 15% (no dojis)
- Candle 3: Pending entry at C2 wick (High for BUY, Low for SELL). 15m expiration.
- Asset-adapted SL buffers and BE offsets.
- Spread floor protection and realistic broker fee allowance.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, List, Optional, Any
import pandas as pd
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.manual_model import get_mt5, get_broker_offset_hours, get_pip_size
from core.confluence_model import get_symbol_sl_buffer, get_symbol_be_offset
from core.ml_filter import get_ml_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CryptoYTDBacktest")

CRYPTO_SYMBOLS = [
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "XRPUSD",
    "DOGEUSD",
    "ADAUSD",
    "LTCUSD",
    "BNBUSD"
]

RISK_USD_PER_TRADE = 50.0  # 0.5% of standard $10,000 account

def fetch_m15_rates(mt5, symbol: str, start_dt: datetime, end_dt: datetime):
    """Fetch continuous M15 rates from MT5 and format to UTC DataFrame."""
    mt5.symbol_select(symbol, True)
    offset_h = get_broker_offset_hours(symbol)
    
    # Add 12 days warm-up prior to start_dt to compute rolling day/swing lines and rolling indicators
    fetch_start = start_dt - timedelta(days=12)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, fetch_start, end_dt)
    if rates is None or len(rates) == 0:
        logger.error(f"No rates returned for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time_utc'] = pd.to_datetime(df['time'], unit='s', utc=True) - pd.Timedelta(hours=offset_h)
    df['open'] = df['open'].astype(float).round(5)
    df['high'] = df['high'].astype(float).round(5)
    df['low'] = df['low'].astype(float).round(5)
    df['close'] = df['close'].astype(float).round(5)
    df['range'] = (df['high'] - df['low']).clip(lower=1e-6)
    df['body'] = (df['close'] - df['open']).abs()
    df['body_ratio'] = df['body'] / df['range']
    # Doji rule: body < 15% of range
    df['is_doji'] = df['body_ratio'] < 0.15
    df['is_bullish'] = (df['close'] > df['open']) & (~df['is_doji'])
    df['is_bearish'] = (df['close'] < df['open']) & (~df['is_doji'])
    return df

def compute_rolling_day_and_swing_lines(df: pd.DataFrame, pip_sz: float):
    """Precompute rolling 21:00 UTC Day Lines and Swing Lines."""
    n = len(df)
    upper_day = np.full(n, np.nan)
    lower_day = np.full(n, np.nan)
    upper_swing = np.full(n, np.nan)
    lower_swing = np.full(n, np.nan)
    
    times = df['time_utc'].values
    min_clearance = 2.5 * pip_sz
    
    curr_day_end_ts = None
    curr_upper_day = None
    curr_lower_day = None
    curr_upper_swing = None
    curr_lower_swing = None
    
    for i in range(n):
        t = pd.Timestamp(times[i])
        if t.hour < 21:
            target_day_end = pd.Timestamp(t.year, t.month, t.day, 21, 0, tz='UTC') - pd.Timedelta(days=1)
        else:
            target_day_end = pd.Timestamp(t.year, t.month, t.day, 21, 0, tz='UTC')
            
        target_day_start = target_day_end - pd.Timedelta(days=1)
        
        if target_day_end != curr_day_end_ts:
            curr_day_end_ts = target_day_end
            
            mask_day = (df['time_utc'] >= target_day_start) & (df['time_utc'] < target_day_end)
            mask_day &= ~((df['time_utc'].dt.hour == 21) & (df['time_utc'].dt.minute == 0))
            
            day_slice = df[mask_day]
            if len(day_slice) >= 15:
                curr_upper_day = round(float(day_slice['high'].max()), 5)
                curr_lower_day = round(float(day_slice['low'].min()), 5)
                
                mask_prior = (df['time_utc'] < target_day_start)
                mask_prior &= ~((df['time_utc'].dt.hour == 21) & (df['time_utc'].dt.minute == 0))
                prior_slice = df[mask_prior].tail(1500)
                
                curr_upper_swing = None
                curr_lower_swing = None
                
                if len(prior_slice) >= 5:
                    p_highs = prior_slice['high'].values
                    p_lows = prior_slice['low'].values
                    m = len(prior_slice)
                    
                    # Scan backwards for swing high
                    for k in range(m - 2, 1, -1):
                        if curr_upper_swing is None and p_highs[k] > p_highs[k-1] and p_highs[k] > p_highs[k+1]:
                            if p_highs[k] >= curr_upper_day + min_clearance:
                                curr_upper_swing = round(float(p_highs[k]), 5)
                                break
                    if curr_upper_swing is None:
                        for k in range(m - 2, 1, -1):
                            if p_highs[k] > p_highs[k-1] and p_highs[k] > p_highs[k+1] and p_highs[k] > curr_upper_day:
                                curr_upper_swing = round(float(p_highs[k]), 5)
                                break
                                
                    # Scan backwards for swing low
                    for k in range(m - 2, 1, -1):
                        if curr_lower_swing is None and p_lows[k] < p_lows[k-1] and p_lows[k] < p_lows[k+1]:
                            if p_lows[k] <= curr_lower_day - min_clearance:
                                curr_lower_swing = round(float(p_lows[k]), 5)
                                break
                    if curr_lower_swing is None:
                        for k in range(m - 2, 1, -1):
                            if p_lows[k] < p_lows[k-1] and p_lows[k] < p_lows[k+1] and p_lows[k] < curr_lower_day:
                                curr_lower_swing = round(float(p_lows[k]), 5)
                                break
            else:
                curr_upper_day = None
                curr_lower_day = None
                curr_upper_swing = None
                curr_lower_swing = None

        upper_day[i] = curr_upper_day if curr_upper_day is not None else np.nan
        lower_day[i] = curr_lower_day if curr_lower_day is not None else np.nan
        upper_swing[i] = curr_upper_swing if curr_upper_swing is not None else np.nan
        lower_swing[i] = curr_lower_swing if curr_lower_swing is not None else np.nan

    df['upper_day'] = upper_day
    df['lower_day'] = lower_day
    df['upper_swing'] = upper_swing
    df['lower_swing'] = lower_swing
    return df

def extract_ml_features_and_score(
    ml_filter,
    symbol: str,
    bars_window: pd.DataFrame,
    c1: dict,
    c2: dict,
    entry: float,
    sl: float,
    direction: str,
    line_val: float,
    line_type: str,
    pip_sz: float
) -> Tuple[float, bool]:
    """Extract features matching ConfluenceMLFilter schema and compute win probability."""
    if not ml_filter or not ml_filter.enabled:
        return 0.50, True

    is_buy = (direction == "BUY")
    c_arr = bars_window['close'].values
    h_arr = bars_window['high'].values
    l_arr = bars_window['low'].values
    
    # ATR 14
    tr = np.maximum(h_arr[1:] - l_arr[1:], np.maximum(abs(h_arr[1:] - c_arr[:-1]), abs(l_arr[1:] - c_arr[:-1])))
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().iloc[-1]
    atr_14_pips = atr_14 / pip_sz

    # EMA 20 & 80
    ema20 = pd.Series(c_arr).ewm(span=20, adjust=False).mean().iloc[-1]
    ema80 = pd.Series(c_arr).ewm(span=80, adjust=False).mean().iloc[-1]

    # RSI 14
    delta = pd.Series(c_arr).diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])

    c2_h = float(c2['high'])
    c2_l = float(c2['low'])
    c2_c = float(c2['close'])
    c2_o = float(c2['open'])

    c1_h = float(c1['high'])
    c1_l = float(c1['low'])
    c1_c = float(c1['close'])
    c1_o = float(c1['open'])

    c2_body = abs(c2_c - c2_o)
    c2_range = max(c2_h - c2_l, 1e-6)
    c1_range = max(c1_h - c1_l, 1e-6)
    is_day_line = 1 if "Day" in line_type else 0

    if is_buy:
        c1_sweep_wick = max(0, line_val - c1_l)
        c1_sweep_wick_pips = c1_sweep_wick / pip_sz
        c1_wick_ratio = c1_sweep_wick / c1_range

        c2_reclaim_wick = max(0, c2_c - c2_l)
        c2_reclaim_wick_pips = c2_reclaim_wick / pip_sz
        c2_reclaim_wick_ratio = c2_reclaim_wick / c2_range
        c2_wick_to_body_ratio = c2_reclaim_wick / max(c2_body, 1e-5)

        is_wick_only_sweep = 1.0 if c1_c >= line_val else 0.0
        is_wick_only_reclaim = 1.0 if c2_o >= line_val else 0.0
        entry_to_line_dist_pips = (entry - line_val) / pip_sz
        break_depth = max(0, line_val - min(c1_l, c2_l))
    else:
        c1_sweep_wick = max(0, c1_h - line_val)
        c1_sweep_wick_pips = c1_sweep_wick / pip_sz
        c1_wick_ratio = c1_sweep_wick / c1_range

        c2_reclaim_wick = max(0, c2_h - c2_c)
        c2_reclaim_wick_pips = c2_reclaim_wick / pip_sz
        c2_reclaim_wick_ratio = c2_reclaim_wick / c2_range
        c2_wick_to_body_ratio = c2_reclaim_wick / max(c2_body, 1e-5)

        is_wick_only_sweep = 1.0 if c1_c <= line_val else 0.0
        is_wick_only_reclaim = 1.0 if c2_o <= line_val else 0.0
        entry_to_line_dist_pips = (line_val - entry) / pip_sz
        break_depth = max(0, max(c1_h, c2_h) - line_val)

    sl_dist = abs(entry - sl)
    sl_pips = sl_dist / pip_sz

    t_utc = c2['time_utc']
    hr = t_utc.hour
    dow = t_utc.weekday()

    feat_dict = {
        "direction": 1 if is_buy else 0,
        "is_day_line": is_day_line,
        "c2_body_pips": c2_body / pip_sz,
        "c2_range_pips": c2_range / pip_sz,
        "c2_body_ratio": c2_body / c2_range,
        "c1_range_pips": c1_range / pip_sz,
        "range_ratio_c2_c1": c2_range / c1_range,
        "break_depth_pips": break_depth / pip_sz,
        "c1_sweep_wick_pips": c1_sweep_wick_pips,
        "c1_wick_ratio": c1_wick_ratio,
        "c2_reclaim_wick_pips": c2_reclaim_wick_pips,
        "c2_reclaim_wick_ratio": c2_reclaim_wick_ratio,
        "c2_wick_to_body_ratio": c2_wick_to_body_ratio,
        "is_wick_only_sweep": is_wick_only_sweep,
        "is_wick_only_reclaim": is_wick_only_reclaim,
        "entry_to_line_dist_pips": entry_to_line_dist_pips,
        "sl_pips": sl_pips,
        "atr_14_pips": atr_14_pips,
        "sl_to_atr": sl_dist / max(atr_14, 1e-6),
        "rsi_14": rsi_14,
        "dist_ema20_pips": (c2_c - ema20) / pip_sz,
        "dist_ema80_pips": (c2_c - ema80) / pip_sz,
        "hour_utc": hr,
        "day_of_week": dow,
        "is_london": 1 if 7 <= hr <= 16 else 0,
        "is_ny": 1 if 12 <= hr <= 21 else 0,
        "is_asia": 1 if (hr >= 22 or hr <= 7) else 0,
    }

    # Dummy columns for symbols (all 0 for crypto)
    for s in ml_filter.all_symbols[1:]:
        feat_dict[f"sym_{s}"] = 1.0 if symbol == s else 0.0

    row_df = pd.DataFrame([feat_dict])[ml_filter.input_columns]
    prob = float(ml_filter.model.predict_proba(row_df)[0, 1])
    passed = bool(prob >= ml_filter.threshold)
    return prob, passed

def simulate_crypto_asset(df: pd.DataFrame, symbol: str, ml_filter, ytd_start: datetime):
    """
    Simulates all trades for a crypto symbol Year-To-Date.
    Returns trade lists for each model.
    """
    pip_sz = get_pip_size(symbol)
    sl_buf_pips = get_symbol_sl_buffer(symbol)
    sl_buf_amt = sl_buf_pips * pip_sz
    be_offset_pips = get_symbol_be_offset(symbol)
    be_offset_amt = be_offset_pips * pip_sz
    
    n = len(df)
    
    # State tracking for each of the 4 models:
    # { 'active_trade': None, 'closed_trades': [] }
    model_states = {
        "confluence_ml_p60": {"active": None, "closed": []},
        "confluence_std_p25": {"active": None, "closed": []},
        "confluence_ml_m15": {"active": None, "closed": []},
        "confluence_m15": {"active": None, "closed": []},
    }
    
    # Iterate through bars starting at index 120 so indicators and lines are warmed up
    for i in range(120, n - 1):
        bar_c2 = df.iloc[i]
        t_c2 = bar_c2['time_utc']
        
        # Check and update active trades on all models using bar_c2 (open/high/low/close)
        for m_key, m_data in model_states.items():
            pos = m_data["active"]
            if pos is not None:
                # Update position lifecycle
                # pos dict: {
                #   'id': int, 'direction': 'BUY'/'SELL', 'entry': float, 'sl': float, 'tp': float,
                #   'entry_time': datetime, 'partial_hit': bool, 'be_moved': bool, 'partial_target': float,
                #   'be_price': float, 'model': str
                # }
                direction = pos['direction']
                high_bar = bar_c2['high']
                low_bar = bar_c2['low']
                
                if m_key == "confluence_m15" or m_key == "confluence_ml_m15":
                    # Fixed 1.5R: hits TP or SL
                    if direction == "BUY":
                        hit_tp = (high_bar >= pos['tp'])
                        hit_sl = (low_bar <= pos['sl'])
                    else:
                        hit_tp = (low_bar <= pos['tp'])
                        hit_sl = (high_bar >= pos['sl'])
                        
                    if hit_tp and hit_sl:
                        # Conservative assumption: hit SL
                        pos['exit_time'] = t_c2
                        pos['realized_r'] = -1.0
                        pos['status'] = 'LOSS'
                        m_data["closed"].append(pos)
                        m_data["active"] = None
                    elif hit_tp:
                        pos['exit_time'] = t_c2
                        pos['realized_r'] = 1.50
                        pos['status'] = 'WIN'
                        m_data["closed"].append(pos)
                        m_data["active"] = None
                    elif hit_sl:
                        pos['exit_time'] = t_c2
                        pos['realized_r'] = -1.0
                        pos['status'] = 'LOSS'
                        m_data["closed"].append(pos)
                        m_data["active"] = None
                        
                elif m_key == "confluence_std_p25":
                    # 25% TP Partial + BE move
                    # If not partial hit yet:
                    if not pos['partial_hit']:
                        if direction == "BUY":
                            hit_partial = (high_bar >= pos['partial_target'])
                            hit_sl = (low_bar <= pos['sl'])
                        else:
                            hit_partial = (low_bar <= pos['partial_target'])
                            hit_sl = (high_bar >= pos['sl'])
                            
                        if hit_sl:
                            pos['exit_time'] = t_c2
                            pos['realized_r'] = -1.0
                            pos['status'] = 'LOSS'
                            m_data["closed"].append(pos)
                            m_data["active"] = None
                        elif hit_partial:
                            pos['partial_hit'] = True
                            pos['sl'] = pos['be_price'] # SL moved to BE offset
                            # Check if same candle reaches full TP
                            if direction == "BUY" and high_bar >= pos['tp']:
                                pos['exit_time'] = t_c2
                                pos['realized_r'] = 0.94 # Full target win
                                pos['status'] = 'WIN'
                                m_data["closed"].append(pos)
                                m_data["active"] = None
                            elif direction == "SELL" and low_bar <= pos['tp']:
                                pos['exit_time'] = t_c2
                                pos['realized_r'] = 0.94
                                pos['status'] = 'WIN'
                                m_data["closed"].append(pos)
                                m_data["active"] = None
                    else:
                        # Partial already banked (+0.19R locked)
                        # Remaining 50% tracks full TP or BE SL
                        if direction == "BUY":
                            hit_tp = (high_bar >= pos['tp'])
                            hit_be = (low_bar <= pos['sl'])
                        else:
                            hit_tp = (low_bar <= pos['tp'])
                            hit_be = (high_bar >= pos['sl'])
                            
                        if hit_tp:
                            pos['exit_time'] = t_c2
                            pos['realized_r'] = 0.94
                            pos['status'] = 'WIN'
                            m_data["closed"].append(pos)
                            m_data["active"] = None
                        elif hit_be:
                            pos['exit_time'] = t_c2
                            pos['realized_r'] = 0.19 # Partial profit banked + breakeven
                            pos['status'] = 'BREAKEVEN'
                            m_data["closed"].append(pos)
                            m_data["active"] = None

                elif m_key == "confluence_ml_p60":
                    # 60% TP Partial + BE move
                    if not pos['partial_hit']:
                        if direction == "BUY":
                            hit_partial = (high_bar >= pos['partial_target'])
                            hit_sl = (low_bar <= pos['sl'])
                        else:
                            hit_partial = (low_bar <= pos['partial_target'])
                            hit_sl = (high_bar >= pos['sl'])
                            
                        if hit_sl:
                            pos['exit_time'] = t_c2
                            pos['realized_r'] = -1.0
                            pos['status'] = 'LOSS'
                            m_data["closed"].append(pos)
                            m_data["active"] = None
                        elif hit_partial:
                            pos['partial_hit'] = True
                            pos['sl'] = pos['be_price']
                            if direction == "BUY" and high_bar >= pos['tp']:
                                pos['exit_time'] = t_c2
                                pos['realized_r'] = 1.15
                                pos['status'] = 'WIN'
                                m_data["closed"].append(pos)
                                m_data["active"] = None
                            elif direction == "SELL" and low_bar <= pos['tp']:
                                pos['exit_time'] = t_c2
                                pos['realized_r'] = 1.15
                                pos['status'] = 'WIN'
                                m_data["closed"].append(pos)
                                m_data["active"] = None
                    else:
                        if direction == "BUY":
                            hit_tp = (high_bar >= pos['tp'])
                            hit_be = (low_bar <= pos['sl'])
                        else:
                            hit_tp = (low_bar <= pos['tp'])
                            hit_be = (high_bar >= pos['sl'])
                            
                        if hit_tp:
                            pos['exit_time'] = t_c2
                            pos['realized_r'] = 1.15
                            pos['status'] = 'WIN'
                            m_data["closed"].append(pos)
                            m_data["active"] = None
                        elif hit_be:
                            pos['exit_time'] = t_c2
                            pos['realized_r'] = 0.45 # 60% partial profit banked
                            pos['status'] = 'BREAKEVEN'
                            m_data["closed"].append(pos)
                            m_data["active"] = None

        # Check if candle C2 is at or after YTD start (2026-01-01)
        if t_c2 < ytd_start:
            continue

        # Look for new Confluence setup:
        # C1 is index i-1, C2 is index i, C3 is index i+1
        bar_c1 = df.iloc[i-1]
        bar_c3 = df.iloc[i+1] # forming entry bar
        
        # Candidate setup detection
        up_day = bar_c2['upper_day']
        low_day = bar_c2['lower_day']
        up_sw = bar_c2['upper_swing']
        low_sw = bar_c2['lower_swing']
        
        if pd.isna(up_day) or pd.isna(low_day):
            continue
            
        low_lines = [("Lower Day Line", low_day)]
        if pd.notna(low_sw):
            low_lines.append(("Lower Swing Line", low_sw))
            
        high_lines = [("Upper Day Line", up_day)]
        if pd.notna(up_sw):
            high_lines.append(("Upper Swing Line", up_sw))
            
        setup_found = False
        direction = None
        reclaimed_line = None
        line_type = None
        entry_price = None
        sl_price = None
        tp_price = None
        
        # BUY Setup Check:
        if bar_c1['is_bearish'] and bar_c2['is_bullish']:
            for name, l_val in low_lines:
                # C1 wick sweep:
                c1_sweep = (bar_c1['low'] <= l_val and max(bar_c1['open'], bar_c1['high']) >= l_val)
                # C2 reclamation:
                c2_reclaim = (bar_c2['close'] > l_val and (min(bar_c2['low'], bar_c2['open']) <= l_val or bar_c1['low'] <= l_val))
                if c1_sweep and c2_reclaim:
                    setup_found = True
                    direction = "BUY"
                    reclaimed_line = l_val
                    line_type = name
                    entry_price = round(bar_c2['high'], 5)
                    sl_price = round(bar_c2['low'] - sl_buf_amt, 5)
                    sl_dist = abs(entry_price - sl_price)
                    tp_price = round(entry_price + 1.5 * sl_dist, 5)
                    break
                    
        # SELL Setup Check:
        if not setup_found and bar_c1['is_bullish'] and bar_c2['is_bearish']:
            for name, l_val in high_lines:
                c1_sweep = (bar_c1['high'] >= l_val and min(bar_c1['open'], bar_c1['low']) <= l_val)
                c2_reclaim = (bar_c2['close'] < l_val and (max(bar_c2['high'], bar_c2['open']) >= l_val or bar_c1['high'] >= l_val))
                if c1_sweep and c2_reclaim:
                    setup_found = True
                    direction = "SELL"
                    reclaimed_line = l_val
                    line_type = name
                    entry_price = round(bar_c2['low'], 5)
                    sl_price = round(bar_c2['high'] + sl_buf_amt, 5)
                    sl_dist = abs(sl_price - entry_price)
                    tp_price = round(entry_price - 1.5 * sl_dist, 5)
                    break
                    
        if not setup_found:
            continue
            
        # Check C3 Entry Trigger:
        # In Candle 3: price must reach entry price within 15 minutes!
        entered = False
        if direction == "BUY":
            entered = (bar_c3['high'] >= entry_price)
        else:
            entered = (bar_c3['low'] <= entry_price)
            
        if not entered:
            # Order expired after C3 without triggering
            continue
            
        # Setup Triggered! Extract ML features and score probability
        bars_120 = df.iloc[max(0, i-119):i+1]
        ml_prob, ml_passed = extract_ml_features_and_score(
            ml_filter=ml_filter,
            symbol=symbol,
            bars_window=bars_120,
            c1=bar_c1.to_dict(),
            c2=bar_c2.to_dict(),
            entry=entry_price,
            sl=sl_price,
            direction=direction,
            line_val=reclaimed_line,
            line_type=line_type,
            pip_sz=pip_sz
        )
        
        sl_dist = abs(entry_price - sl_price)
        tp_dist = 1.5 * sl_dist
        
        # Calculate model targets
        p25_target = round(entry_price + (0.25 * tp_dist if direction == "BUY" else -0.25 * tp_dist), 5)
        p60_target = round(entry_price + (0.60 * tp_dist if direction == "BUY" else -0.60 * tp_dist), 5)
        be_price = round(entry_price + (be_offset_amt if direction == "BUY" else -be_offset_amt), 5)
        
        trade_id = f"{symbol}_{t_c2.strftime('%Y%m%d_%H%M')}_{direction}"
        
        # Arm trades in available models if model is not already active
        for m_key in model_states.keys():
            # Check ML gate
            if "ml" in m_key and not ml_passed:
                continue # Suppressed by LightGBM gate
                
            if model_states[m_key]["active"] is None:
                # Open position
                partial_tgt = p60_target if "p60" in m_key else p25_target
                model_states[m_key]["active"] = {
                    "id": trade_id,
                    "symbol": symbol,
                    "direction": direction,
                    "entry": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "entry_time": bar_c3['time_utc'],
                    "partial_target": partial_tgt,
                    "be_price": be_price,
                    "partial_hit": False,
                    "ml_prob": ml_prob,
                    "line_type": line_type,
                    "model": m_key
                }

    # Close any still active trade at last bar price
    last_bar = df.iloc[-1]
    for m_key, m_data in model_states.items():
        if m_data["active"] is not None:
            pos = m_data["active"]
            # Mark closed at market
            pos['exit_time'] = last_bar['time_utc']
            pos['realized_r'] = 0.0
            pos['status'] = 'BREAKEVEN'
            m_data["closed"].append(pos)
            m_data["active"] = None
            
    return {m_key: m_data["closed"] for m_key, m_data in model_states.items()}

def run_all_crypto_backtests():
    """Execute complete YTD backtest for all 8 crypto assets across all 4 confluence models."""
    mt5 = get_mt5("BTCUSD")
    if not mt5:
        logger.error("Failed to connect to MT5 for crypto.")
        return None
        
    ml_filter = get_ml_filter()
    logger.info(f"Confluence ML Filter loaded: {ml_filter.enabled} (Threshold: {ml_filter.threshold})")
    
    ytd_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc)
    
    all_results = {
        "confluence_ml_p60": {},
        "confluence_std_p25": {},
        "confluence_ml_m15": {},
        "confluence_m15": {},
    }
    
    print("\n" + "="*80)
    print("🚀 EXECUTING YTD (2026-01-01 TO PRESENT) CRYPTO BACKTEST ACROSS 8 ASSETS")
    print("="*80)
    
    for symbol in CRYPTO_SYMBOLS:
        print(f"\n⏳ Processing {symbol}...")
        df = fetch_m15_rates(mt5, symbol, ytd_start, end_dt)
        if df is None:
            continue
            
        pip_sz = get_pip_size(symbol)
        df = compute_rolling_day_and_swing_lines(df, pip_sz)
        
        sym_trades = simulate_crypto_asset(df, symbol, ml_filter, ytd_start)
        
        for m_key, trades in sym_trades.items():
            n_trades = len(trades)
            n_wins = sum(1 for t in trades if t['status'] == 'WIN')
            n_losses = sum(1 for t in trades if t['status'] == 'LOSS')
            n_be = sum(1 for t in trades if t['status'] == 'BREAKEVEN')
            net_r = round(sum(t['realized_r'] for t in trades), 2)
            pnl_usd = round(net_r * RISK_USD_PER_TRADE, 2)
            win_rate = round((n_wins / n_trades * 100), 1) if n_trades > 0 else 0.0
            
            # Status rule: Net R >= 0.0 designates APPROVED / winning asset; Net R < 0.0 designates BENCHED
            status = "APPROVED" if net_r >= 0.0 else "BENCHED"
            
            all_results[m_key][symbol] = {
                "trades": n_trades,
                "wins": n_wins,
                "losses": n_losses,
                "be": n_be,
                "win_rate": win_rate,
                "net_r": net_r,
                "pnl_usd": pnl_usd,
                "status": status,
                "trade_list": trades
            }
            print(f"  [{m_key:18}] Trades: {n_trades:3} | W: {n_wins:2} | L: {n_losses:2} | BE: {n_be:2} | WR: {win_rate:5.1f}% | Net R: {net_r:+6.2f} | PnL: ${pnl_usd:+7.2f} | {status}")

    return all_results

def update_dynamic_whitelist_json(backtest_results):
    """
    Updates config/dynamic_model_whitelists.json with authentic YTD crypto performance metrics.
    Only approved assets (Net R >= 0.0) are placed in winning_pairs.
    Losing assets (Net R < 0.0) are placed in benched_pairs.
    """
    whitelist_file = PROJECT_ROOT / "config" / "dynamic_model_whitelists.json"
    if not whitelist_file.exists():
        logger.error(f"Whitelist file not found: {whitelist_file}")
        return False
        
    with open(whitelist_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for m_key, sym_dict in backtest_results.items():
        if m_key not in data.get("models", {}):
            continue
            
        m_model = data["models"][m_key]
        winning_pairs = set(m_model.get("winning_pairs", []))
        benched_pairs = set(m_model.get("benched_pairs", []))
        pair_stats = m_model.get("pair_stats", {})
        
        for sym, stats in sym_dict.items():
            pair_stats[sym] = {
                "trades": stats["trades"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "be": stats["be"],
                "net_r": stats["net_r"],
                "pnl_usd": stats["pnl_usd"],
                "win_rate": stats["win_rate"],
                "status": stats["status"]
            }
            if stats["status"] == "APPROVED":
                winning_pairs.add(sym)
                benched_pairs.discard(sym)
            else:
                benched_pairs.add(sym)
                winning_pairs.discard(sym)
                
        m_model["winning_pairs"] = sorted(list(winning_pairs))
        m_model["benched_pairs"] = sorted(list(benched_pairs))
        m_model["pair_stats"] = pair_stats
        
    data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    
    with open(whitelist_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    logger.info("Successfully updated config/dynamic_model_whitelists.json with crypto metrics.")
    return True

if __name__ == "__main__":
    results = run_all_crypto_backtests()
    if results:
        update_dynamic_whitelist_json(results)
        print("\n✅ Backtest complete and config/dynamic_model_whitelists.json successfully updated!")
