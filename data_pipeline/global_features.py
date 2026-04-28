import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class GlobalFeatureEngineer:
    """
    Computes global market features across multiple currency pairs.
    
    Features:
    - Currency Strength Matrix (CSM): Relative strength of USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD.
    - Synthetic DXY: Proxy for Dollar Index.
    - VIX Proxy: Realised 24-hour volatility of EURUSD as a fear gauge.
    - Yield Curve Slope: 10Y - 2Y US Treasury spread (recession/risk signal).
    """
    
    CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    
    def __init__(self):
        pass

    def compute_currency_strength(self, aligned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates the relative strength of each major currency.
        
        Algorithm:
        1. Calculate hourly returns for every pair.
        2. Decomposition: If EURUSD goes up, EUR is strong (+1) and USD is weak (-1).
        3. Average the relative strength of each currency against all others.
        
        Args:
            aligned_data: Dict mapping 'pair_name' to aligned OHLCV DataFrames.
            
        Returns:
            DataFrame with columns [USD_strength, EUR_strength, ...]
        """
        # 1. Extract log returns for all pairs
        all_returns = {}
        for pair, df in aligned_data.items():
            if len(df) < 2: continue
            all_returns[pair] = np.log(df['close'] / df['close'].shift(1))
            
        returns_df = pd.DataFrame(all_returns).ffill().fillna(0)
        
        # 2. Score each currency
        strength = pd.DataFrame(0.0, index=returns_df.index, columns=self.CURRENCIES)
        
        for pair in returns_df.columns:
            # Standard pairs like EURUSD
            base = pair[:3]
            quote = pair[3:6]
            
            if base in self.CURRENCIES and quote in self.CURRENCIES:
                ret = returns_df[pair]
                strength[base] += ret
                strength[quote] -= ret
        
        # Normalize: Average by number of appearances (usually 7 for majors)
        # Actually, just use standard rolling z-score for normalization later in trainer
        return strength

    def compute_dxy_proxy(self, aligned_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculates a synthetic Dollar Index (DXY) proxy using G7 weights.
        
        Approximate DXY Weights:
        EUR: 57.6%, JPY: 13.6%, GBP: 11.9%, CAD: 9.1%, SEK: 4.2%, CHF: 3.6%
        (We'll exclude SEK and re-weight G5 + CHF)
        """
        weights = {
            "EURUSD": 0.576,
            "USDJPY": 0.136, # Note USD is base for JPY in DXY formula, but we usually have USDJPY
            "GBPUSD": 0.119,
            "USDCAD": 0.091,
            "USDCHF": 0.036
        }
        
        dxy_log = pd.Series(0.0, index=next(iter(aligned_data.values())).index)
        
        # Log-DXY approx = sum(weight * log(USD_relative_price))
        # For EURUSD (EUR is base): USD relative price = 1 / EURUSD
        # For USDCAD (USD is base): USD relative price = USDCAD
        
        for pair, weight in weights.items():
            if pair not in aligned_data: continue
            df = aligned_data[pair]
            
            if pair.startswith("USD"):
                dxy_log += weight * np.log(df['close'])
            else:
                dxy_log -= weight * np.log(df['close'])
                
        return np.exp(dxy_log)

    def compute_vix_proxy(self, pair_df: pd.DataFrame, window: int = 24) -> pd.Series:
        """
        Synthetic VIX: 24-hour rolling realised volatility of EURUSD (or any anchor pair).
        Annualised to match VIX convention (multiply by sqrt(6240) for hourly, 252-day year).
        
        A z-score is returned so the model sees relative fear, not absolute vol.
        """
        log_ret = np.log(pair_df['close'] / pair_df['close'].shift(1))
        realised_vol = log_ret.rolling(window).std() * np.sqrt(6240)  # annualised
        # Z-score over 30-day window
        mu  = realised_vol.rolling(720).mean()
        sig = realised_vol.rolling(720).std()
        vix_z = ((realised_vol - mu) / sig.replace(0, np.nan)).fillna(0)
        return vix_z

    def compute_yield_curve_slope(self, tnx_df: pd.DataFrame,
                                   tlt_df: pd.DataFrame = None) -> pd.Series:
        """
        Yield Curve Slope proxy.
        
        If 2Y Treasury ('^IRX') is available, use 10Y - 2Y.
        Otherwise, use a rate-of-change proxy on the 10Y (TNX): a flattening/inversion
        indicator derived from 10Y momentum vs 1Y lookback.
        
        Negative slope = inverted curve = recession risk = CRISIS territory.
        """
        if tlt_df is not None and not tlt_df.empty:
            # 10Y - 2Y spread (both in % terms)
            slope = tnx_df['close'] - tlt_df['close']
        else:
            # Proxy: 10Y rate vs its 252-bar (1 year) moving average
            # Inverted when current rate < long-run average (dovish pivot incoming)
            ma252 = tnx_df['close'].rolling(252).mean()
            slope = tnx_df['close'] - ma252
        
        # Normalise
        slope_z = ((slope - slope.rolling(720).mean()) /
                   slope.rolling(720).std().replace(0, np.nan)).fillna(0)
        return slope_z

    def add_global_features(self, symbol: str, pair_features: pd.DataFrame, aligned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Enriches a single pair's features with global context.
        Reduced to 2 features to match 34-feature brain (32 base + 2 global).
        """
        enriched = pair_features.copy()
        
        # 1. Add Gold Return (Critical for Volatility Perception)
        if "GOLD" in aligned_data:
            gold_df = aligned_data["GOLD"]
            enriched['gold_ret'] = np.log(gold_df['close'] / gold_df['close'].shift(1)).fillna(0)
        else:
            enriched['gold_ret'] = 0.0
            
        # 2. Add DXY Proxy (Critical for USD Sensitivity)
        try:
            dxy = self.compute_dxy_proxy(aligned_data)
            enriched['dxy_ret'] = np.log(dxy / dxy.shift(1)).fillna(0)
        except:
            enriched['dxy_ret'] = 0.0
            
        return enriched.ffill().fillna(0)
