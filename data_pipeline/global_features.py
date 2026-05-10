import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class GlobalFeatureEngineer:
    """
    Computes global market features across multiple currency pairs.

    Features (v2 — 16 total):
    - Currency Strength Matrix (CSM): Relative strength of USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD. (8)
    - Synthetic DXY: Proxy for Dollar Index. (2)
    - Gold Return: Risk-off barometer. (1)
    - VIX Proxy: Realised 24-hour volatility of EURUSD as a fear gauge. (1)
    - Yield Curve Slope: 10Y - 2Y US Treasury spread (recession/risk signal). (1)
    - S&P 500 Return: Global equity risk appetite. (1)     [NEW v2]
    - Crude Oil Return: Commodity inflation / CAD/AUD driver. (1)  [NEW v2]
    - NASDAQ Return: Tech-sector / USD liquidity proxy. (1)  [NEW v2]
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
        Enriches a single pair's features with full global context (13 features).
        """
        enriched = pair_features.copy()
        
        # 1. Currency Strength Matrix (8 features)
        try:
            csm = self.compute_currency_strength(aligned_data)
            for curr in self.CURRENCIES:
                col_name = f"{curr}_strength"
                if curr in csm.columns:
                    enriched[col_name] = csm[curr]
                else:
                    enriched[col_name] = 0.0
        except Exception as e:
            logger.warning(f"Failed to compute CSM: {e}")
            for curr in self.CURRENCIES: enriched[f"{curr}_strength"] = 0.0

        # 2. DXY Proxy & Returns (2 features)
        try:
            dxy = self.compute_dxy_proxy(aligned_data)
            enriched['dxy_proxy'] = dxy
            enriched['dxy_ret'] = np.log(dxy / dxy.shift(1)).fillna(0)
        except:
            enriched['dxy_proxy'] = 100.0
            enriched['dxy_ret'] = 0.0

        # 3. Gold Return (1 feature)
        if "GOLD" in aligned_data:
            gold_df = aligned_data["GOLD"]
            enriched['gold_ret'] = np.log(gold_df['close'] / gold_df['close'].shift(1)).fillna(0)
        else:
            enriched['gold_ret'] = 0.0

        # 4. VIX Proxy (1 feature)
        # Use EURUSD as the volatility anchor if available, otherwise the current pair
        anchor_df = aligned_data.get("EURUSD", enriched)
        try:
            enriched['vix_proxy'] = self.compute_vix_proxy(anchor_df)
        except:
            enriched['vix_proxy'] = 0.0

        # 5. Yield Curve Slope (1 feature)
        if "^TNX" in aligned_data:
            tnx_df = aligned_data["^TNX"]
            irx_df = aligned_data.get("^IRX") # 2Y Treasury
            try:
                enriched['yield_curve_slope'] = self.compute_yield_curve_slope(tnx_df, irx_df)
            except:
                enriched['yield_curve_slope'] = 0.0
        else:
            enriched['yield_curve_slope'] = 0.0

        # 6. S&P 500 Return (1 feature) [NEW v2]
        if "SP500" in aligned_data:
            sp_df = aligned_data["SP500"]
            try:
                enriched['sp500_ret'] = np.log(sp_df['close'] / sp_df['close'].shift(1)).fillna(0)
            except:
                enriched['sp500_ret'] = 0.0
        else:
            enriched['sp500_ret'] = 0.0

        # 7. Crude Oil Return (1 feature) [NEW v2]
        if "OIL" in aligned_data:
            oil_df = aligned_data["OIL"]
            try:
                enriched['oil_ret'] = np.log(oil_df['close'] / oil_df['close'].shift(1)).fillna(0)
            except:
                enriched['oil_ret'] = 0.0
        else:
            enriched['oil_ret'] = 0.0

        # 8. NASDAQ Return (1 feature) [NEW v2]
        if "NASDAQ" in aligned_data:
            ndx_df = aligned_data["NASDAQ"]
            try:
                enriched['nasdaq_ret'] = np.log(ndx_df['close'] / ndx_df['close'].shift(1)).fillna(0)
            except:
                enriched['nasdaq_ret'] = 0.0
        else:
            enriched['nasdaq_ret'] = 0.0

        return enriched.ffill().fillna(0)
