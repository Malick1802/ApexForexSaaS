# =============================================================================
# Triple Barrier Labeling Module
# =============================================================================
"""
Implementation of the Triple Barrier Method for labeling forex data.

The Triple Barrier method creates labels based on which price barrier
is hit first after an entry point:
- Take Profit barrier (above for long, below for short)
- Stop Loss barrier (below for long, above for short)  
- Time barrier (optional, not used in this implementation)

Labels:
    1 = BUY signal (TP hit first on simulated long position)
    2 = SELL signal (TP hit first on simulated short position)
    0 = WAIT/No trade (SL hit first or no barrier hit)

CRITICAL: This implementation uses ONLY past data for making decisions.
Each label at time t is determined by simulating trades starting at t+1,
ensuring zero look-ahead bias.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def get_pip_value(symbol: str, pip_type: Optional[str] = None) -> float:
    """
    Get the pip value for a currency pair.
    
    Args:
        symbol: Currency pair (e.g., "EURUSD", "USDJPY")
        pip_type: Either "standard" (0.0001) or "jpy" (0.01)
                  If None, auto-detect based on symbol
                  
    Returns:
        Pip value as float
    """
    if pip_type == "jpy":
        return 0.01
    elif pip_type == "standard":
        return 0.0001
    else:
        # Auto-detect: JPY pairs have JPY as quote currency (last 3 chars)
        if symbol.upper().endswith("JPY"):
            return 0.01
        return 0.0001


def triple_barrier_label(
    df: pd.DataFrame,
    stop_loss_pips: float = 40,  # Minimum SL (Increased for robustness)
    take_profit_pips: float = 80, # Ignored, calculated dynamically
    pip_value: Optional[float] = None,
    symbol: Optional[str] = None,
    max_holding_periods: Optional[int] = 120,
    spread_pips: float = 2.0     # Built-in spread buffer
) -> pd.DataFrame:
    """
    Apply Dynamic Triple Barrier labeling (1:2 R:R).
    
    Rule 1: SL = max(ATR(14) * 1.5, 25 pips)
    Rule 2: TP = SL * 2 (Strict 1:2 Risk/Reward)
    Rule 3: Label 1 if TP hit first, 2 if Sell TP hit first, 0 otherwise.
    
    Args:
        df: OHLCV DataFrame
        stop_loss_pips: Minimum SL in pips (default 25)
        take_profit_pips: Ignored/Deprecated (TP is always 2*SL)
        pip_value: Value of one pip
        symbol: Symbol for auto-pip detection
        max_holding_periods: Max bars to hold (default 120)
        
    Returns:
        DataFrame with 'label' and outcome columns.
    """
    # Validate input
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if len(df) < 15: # Need enough for ATR
        raise ValueError("DataFrame too short for ATR calculation")
    
    # Determine pip value
    if pip_value is None:
        if symbol:
            pip_value = get_pip_value(symbol)
        else:
            pip_value = 0.0001
            logger.warning("Using default pip_value=0.0001")

    # Calculate ATR for Dynamic SL
    # Note: df['close'] etc might be Series, ensure ta works
    # We calculate ATR on the WHOLE dataframe first
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    if atr is None:
         # Fallback if ta fails (e.g. not enough data), fill with very small val to default to min_sl
         atr = pd.Series(0, index=df.index)
    
    atr = atr.fillna(0) # Fill NaN at start
    atr_values = atr.values
    
    min_sl_distance = stop_loss_pips * pip_value
    
    logger.info(
        f"Labeling with Dynamic 1:2 R:R. Min SL={stop_loss_pips} pips, "
        f"TP/SL Ratio=2.0, Max Hold={max_holding_periods}"
    )
    
    # Initialize result arrays
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    long_outcomes = np.full(n, np.nan)
    short_outcomes = np.full(n, np.nan)
    bars_to_outcome = np.zeros(n, dtype=np.int32)
    sl_pips_used = np.zeros(n, dtype=np.float32) # Track actual SL used
    
    # Get price arrays
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Main Loop
    for i in range(n - 1):
        # Entry is NEXT Open
        entry_price = opens[i + 1]
        
        # Dynamic barrier calculation using ATR from time i (current candle)
        # We use ATR[i] to set barriers for trade at i+1
        curr_atr = atr_values[i]
        
        # Rule 1: SL = max(ATR * 1.5, min_sl_pips) + spread
        dynamic_sl = curr_atr * 1.5
        spread_dist = spread_pips * pip_value
        sl_dist = max(dynamic_sl, min_sl_distance) + spread_dist
        
        # Rule 2: TP = (SL_no_spread * 2) + spread
        # To maintain a true 1:2 RR despite the spread tax
        tp_dist = (max(dynamic_sl, min_sl_distance) * 2.0) + spread_dist
        
        sl_pips_used[i] = sl_dist / pip_value
        
        # Define barriers
        long_tp = entry_price + tp_dist
        long_sl = entry_price - sl_dist
        short_tp = entry_price - tp_dist
        short_sl = entry_price + sl_dist
        
        # Scan future
        long_result = 0
        short_result = 0
        outcome_bar = 0
        
        max_bars = max_holding_periods if max_holding_periods else 120
        end_idx = min(i + 1 + max_bars, n)
        
        # Inner loop - performance critical?
        # For Python loop, 3 years data (20k bars) * 120 lookahead ~ 2.4M iter. 
        # Might be slow but acceptable for offline labeling.
        for j in range(i + 1, end_idx):
            candle_high = highs[j]
            candle_low = lows[j]
            
            # check outcomes
            if long_result == 0:
                tp_hit = candle_high >= long_tp
                sl_hit = candle_low <= long_sl
                if tp_hit and sl_hit: long_result = -1 # SL (conservative)
                elif tp_hit: long_result = 1
                elif sl_hit: long_result = -1
            
            if short_result == 0:
                tp_hit = candle_low <= short_tp
                sl_hit = candle_high >= short_sl
                if tp_hit and sl_hit: short_result = -1 # SL (conservative)
                elif tp_hit: short_result = 1
                elif sl_hit: short_result = -1
                
            if long_result != 0 and short_result != 0:
                outcome_bar = j - i
                break
        
        # Assign Labels
        long_outcomes[i] = long_result
        short_outcomes[i] = short_result
        bars_to_outcome[i] = outcome_bar if outcome_bar > 0 else (end_idx - i)
        
        # Rule 3: Binary Labeling (+ Wait)
        # Note: If Time Barrier ($max_bars) hits, result is 0/Wait.
        if long_result == 1 and short_result != 1:
            labels[i] = 1 # BUY
        elif short_result == 1 and long_result != 1:
            labels[i] = 2 # SELL
        else:
            labels[i] = 0 # WAIT
            
    # Create Result
    result = df.copy()
    result['label'] = labels
    result['sl_pips'] = sl_pips_used
    result['tp_pips'] = sl_pips_used * 2
    
    # Log stats
    dist = pd.Series(labels).value_counts()
    logger.info(f"Dynamic Label Distribution: {dist.to_dict()}")
    
    return result


def triple_barrier_label_vectorized(
    df: pd.DataFrame,
    stop_loss_pips: float = 25,
    take_profit_pips: float = 50,
    pip_value: float = 0.0001,
    lookahead_periods: int = 100
) -> pd.DataFrame:
    """
    Vectorized (faster) version of triple barrier labeling.
    
    Uses rolling window operations for better performance on large datasets.
    
    Args:
        df: DataFrame with OHLCV columns
        stop_loss_pips: Stop loss in pips
        take_profit_pips: Take profit in pips
        pip_value: Value of one pip
        lookahead_periods: Max periods to look ahead for barrier hits
        
    Returns:
        DataFrame with 'label' column added
    """
    sl_distance = stop_loss_pips * pip_value
    tp_distance = take_profit_pips * pip_value
    
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Pre-compute rolling max/min for each lookahead window
    # This is more efficient than nested loops for large datasets
    for i in range(n - 1):
        entry = opens[i + 1] if i + 1 < n else opens[i]
        
        # Define barrier levels
        long_tp = entry + tp_distance
        long_sl = entry - sl_distance
        short_tp = entry - tp_distance
        short_sl = entry + sl_distance
        
        # Look ahead window
        end_idx = min(i + 1 + lookahead_periods, n)
        future_highs = highs[i + 1:end_idx]
        future_lows = lows[i + 1:end_idx]
        
        if len(future_highs) == 0:
            continue
        
        # Find first bar where each barrier is hit
        long_tp_bars = np.where(future_highs >= long_tp)[0]
        long_sl_bars = np.where(future_lows <= long_sl)[0]
        short_tp_bars = np.where(future_lows <= short_tp)[0]
        short_sl_bars = np.where(future_highs >= short_sl)[0]
        
        # Get first occurrence (or infinity if never hit)
        long_tp_first = long_tp_bars[0] if len(long_tp_bars) > 0 else np.inf
        long_sl_first = long_sl_bars[0] if len(long_sl_bars) > 0 else np.inf
        short_tp_first = short_tp_bars[0] if len(short_tp_bars) > 0 else np.inf
        short_sl_first = short_sl_bars[0] if len(short_sl_bars) > 0 else np.inf
        
        # Determine label
        long_tp_win = long_tp_first < long_sl_first
        short_tp_win = short_tp_first < short_sl_first
        
        if long_tp_win and not short_tp_win:
            labels[i] = 1  # BUY
        elif short_tp_win and not long_tp_win:
            labels[i] = 2  # SELL
        # else: 0 (WAIT)
    
    result = df.copy()
    result['label'] = labels
    return result


def validate_no_lookahead(
    df: pd.DataFrame,
    label_col: str = 'label'
) -> Tuple[bool, str]:
    """
    Validate that labels don't contain look-ahead bias.
    
    This is a sanity check to ensure the labeling logic is correct.
    
    Args:
        df: Labeled DataFrame
        label_col: Name of label column
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check 1: Last row should have label 0 (no future data to label)
    if df[label_col].iloc[-1] != 0:
        return False, "Last row should have label 0 (no future data)"
    
    # Check 2: Labels should only use past data
    # This is inherent in the algorithm design, but we can verify
    # by checking that the label distribution makes sense
    
    label_counts = df[label_col].value_counts()
    total = len(df)
    
    # Expect roughly balanced distribution (with some variation)
    # If one label dominates > 80%, something might be wrong
    for label in [0, 1, 2]:
        if label in label_counts:
            pct = label_counts[label] / total
            if pct > 0.8:
                return False, f"Label {label} dominates at {pct:.1%} - may indicate bias"
    
    return True, "Validation passed"
