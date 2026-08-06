"""
Regime-Conditional Filter Module
================================
Based on arxiv paper "Discovery of a 13-Sharpe OOS Factor" (NASA researcher, Nov 2025)

Key Insight: Signals that appear weak on average become EXTRAORDINARILY powerful
when applied selectively during specific market conditions.

How it works:
- Calculate UpFraction = % of positive days in trailing 63-day window
- Only trade when UpFraction > 0.55 (bullish drift regime)
- Skip trades when UpFraction < 0.45 (bearish regime)
- Signal = BASE × REGIME (binary gate: 0 or 1)

Results from paper:
- Annualized return: 158.6%
- Volatility: 12.0%
- Max drawdown: -11.9%
- Walk-forward validated over 20 years (2004-2024)
- 1,000 randomization trials, p-value < 0.001
- Sharpe > 7 across 30% parameter variations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def calculate_regime(
    df: pd.DataFrame,
    lookback_days: int = 63,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.45,
) -> pd.DataFrame:
    """
    Calculate regime-conditional filter based on drift regime.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data (any timeframe)
    lookback_days : int
        Number of days to calculate UpFraction (default: 63, ~3 months)
    entry_threshold : float
        Minimum UpFraction to enter trades (default: 0.55)
    exit_threshold : float
        Maximum UpFraction before exiting trades (default: 0.45)
    
    Returns
    -------
    DataFrame with regime columns added:
    - 'up_fraction': % of positive days in lookback window
    - 'regime': 'bullish', 'bearish', or 'neutral'
    - 'regime_signal': 1 for bullish, -1 for bearish, 0 for neutral
    """
    df = df.copy()
    
    # Calculate daily returns
    if "close" in df.columns:
        # Resample to daily if intraday
        if hasattr(df.index, 'hour'):
            daily = df.resample("D")["close"].last().dropna()
        else:
            daily = df["close"]
        
        # Calculate daily returns
        daily_returns = daily.pct_change().dropna()
        
        # Calculate UpFraction over lookback window
        up_fraction = daily_returns.rolling(lookback_days).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # Resample back to original timeframe if needed
        if hasattr(df.index, 'hour'):
            up_fraction = up_fraction.reindex(df.index, method="ffill")
        
        df["up_fraction"] = up_fraction
        
        # Classify regime
        df["regime"] = "neutral"
        df.loc[df["up_fraction"] > entry_threshold, "regime"] = "bullish"
        df.loc[df["up_fraction"] < exit_threshold, "regime"] = "bearish"
        
        # Generate regime signal
        df["regime_signal"] = 0
        df.loc[df["regime"] == "bullish", "regime_signal"] = 1
        df.loc[df["regime"] == "bearish", "regime_signal"] = -1
    else:
        df["up_fraction"] = 0.5
        df["regime"] = "neutral"
        df["regime_signal"] = 0
    
    return df


def apply_regime_filter(
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    df: pd.DataFrame,
    lookback_days: int = 63,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.45,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Apply regime-conditional filter to trading signals.
    
    Parameters
    ----------
    long_entries, long_exits, short_entries, short_exits : pd.Series (bool)
        Original trading signals
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to calculate UpFraction
    entry_threshold : float
        Minimum UpFraction to enter trades
    exit_threshold : float
        Maximum UpFraction before exiting trades
    
    Returns
    -------
    Tuple of filtered signals (long_entries, long_exits, short_entries, short_exits)
    """
    # Calculate regime
    df_regime = calculate_regime(df, lookback_days, entry_threshold, exit_threshold)
    
    # Apply regime filter
    # Only allow long entries when regime is bullish
    # Only allow short entries when regime is bearish
    filtered_long_entries = long_entries & (df_regime["regime_signal"] == 1)
    filtered_short_entries = short_entries & (df_regime["regime_signal"] == -1)
    
    # Exits remain unchanged (exit regardless of regime)
    filtered_long_exits = long_exits
    filtered_short_exits = short_exits
    
    return filtered_long_entries, filtered_long_exits, filtered_short_entries, filtered_short_exits


def calculate_regime_strength(
    df: pd.DataFrame,
    lookback_days: int = 63,
) -> pd.Series:
    """
    Calculate regime strength (how strong the trend is).
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to calculate strength
    
    Returns
    -------
    Series with regime strength (0-1, where 1 is strongest trend)
    """
    df = df.copy()
    
    # Calculate daily returns
    if "close" in df.columns:
        # Resample to daily if intraday
        if hasattr(df.index, 'hour'):
            daily = df.resample("D")["close"].last().dropna()
        else:
            daily = df["close"]
        
        # Calculate daily returns
        daily_returns = daily.pct_change().dropna()
        
        # Calculate regime strength (absolute deviation from 0.5)
        # Strong bullish: UpFraction close to 1.0
        # Strong bearish: UpFraction close to 0.0
        # Neutral: UpFraction close to 0.5
        up_fraction = daily_returns.rolling(lookback_days).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # Strength = |UpFraction - 0.5| * 2 (normalized to 0-1)
        strength = (up_fraction - 0.5).abs() * 2
        
        # Resample back to original timeframe if needed
        if hasattr(df.index, 'hour'):
            strength = strength.reindex(df.index, method="ffill")
        
        return strength
    else:
        return pd.Series(0.5, index=df.index)


def regime_summary(df: pd.DataFrame, lookback_days: int = 63) -> dict:
    """
    Generate summary statistics for regime distribution.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to calculate regime
    
    Returns
    -------
    dict with regime statistics
    """
    df_regime = calculate_regime(df, lookback_days)
    
    regime_counts = df_regime["regime"].value_counts()
    total_bars = len(df_regime)
    
    summary = {
        "total_bars": total_bars,
        "bullish_bars": regime_counts.get("bullish", 0),
        "bearish_bars": regime_counts.get("bearish", 0),
        "neutral_bars": regime_counts.get("neutral", 0),
        "bullish_pct": regime_counts.get("bullish", 0) / total_bars * 100,
        "bearish_pct": regime_counts.get("bearish", 0) / total_bars * 100,
        "neutral_pct": regime_counts.get("neutral", 0) / total_bars * 100,
        "avg_up_fraction": df_regime["up_fraction"].mean(),
        "std_up_fraction": df_regime["up_fraction"].std(),
    }
    
    return summary


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/admin1/project9/backtest")
    
    # Load sample data
    DATA = "/mnt/c/Users/Admin/project9/data"
    df = pd.read_parquet(f"{DATA}/NVDA_5min.parquet")
    
    print("=" * 70)
    print("REGIME-CONDITIONAL FILTER ANALYSIS")
    print("=" * 70)
    
    # Calculate regime
    df_regime = calculate_regime(df, lookback_days=63)
    
    # Print summary
    summary = regime_summary(df, lookback_days=63)
    print(f"\nRegime Distribution:")
    print(f"  Bullish: {summary['bullish_pct']:.1f}% ({summary['bullish_bars']} bars)")
    print(f"  Bearish: {summary['bearish_pct']:.1f}% ({summary['bearish_bars']} bars)")
    print(f"  Neutral: {summary['neutral_pct']:.1f}% ({summary['neutral_bars']} bars)")
    print(f"  Avg UpFraction: {summary['avg_up_fraction']:.3f}")
    print(f"  Std UpFraction: {summary['std_up_fraction']:.3f}")
    
    # Calculate regime strength
    strength = calculate_regime_strength(df, lookback_days=63)
    print(f"\nRegime Strength:")
    print(f"  Avg Strength: {strength.mean():.3f}")
    print(f"  Std Strength: {strength.std():.3f}")
    
    print("\n" + "=" * 70)
    print("REGIME FILTER IMPLEMENTATION COMPLETE")
    print("=" * 70)
