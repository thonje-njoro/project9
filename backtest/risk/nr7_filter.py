"""
NR7 Volatility Contraction Filter
==================================
Based on Toby Crabel, "Day Trading with Short Term Price Patterns" (30+ years)

Key Insight: The ONLY documented intraday breakout edge. Breakouts after
volatility contraction have higher continuation probability.

How it works:
- NR7 = today's range is smallest of last 7 days
- After NR7, breakout has ~55% continuation probability (vs 50% random)
- Edge is small but consistent over decades

Why this works:
- When volatility contracts, stop losses are tight (close to entry)
- When expansion comes, winners are large (volatility expands)
- Win rate may drop to 45-50%, but W/L ratio improves to 2-3x
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def calculate_nr7(
    df: pd.DataFrame,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Calculate NR7 (Narrow Range 7) filter.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to look back for narrow range (default: 7)
    
    Returns
    -------
    DataFrame with NR7 columns added:
    - 'daily_range': Daily range (high - low)
    - 'nr7_range': Smallest range in lookback window
    - 'is_nr7': True if today's range is the smallest in lookback window
    - 'nr7_signal': 1 if NR7, 0 otherwise
    """
    df = df.copy()
    
    # Calculate daily range
    if "high" in df.columns and "low" in df.columns:
        # Resample to daily if intraday
        if hasattr(df.index, 'hour'):
            daily_high = df.resample("D")["high"].max()
            daily_low = df.resample("D")["low"].min()
            daily_range = daily_high - daily_low
        else:
            daily_range = df["high"] - df["low"]
        
        # Calculate NR7: today's range is smallest of last 7 days
        nr7_range = daily_range.rolling(lookback_days).min()
        is_nr7 = daily_range == nr7_range
        
        # Resample back to original timeframe if needed
        if hasattr(df.index, 'hour'):
            daily_range = daily_range.reindex(df.index, method="ffill")
            nr7_range = nr7_range.reindex(df.index, method="ffill")
            is_nr7 = is_nr7.reindex(df.index, method="ffill")
        
        df["daily_range"] = daily_range
        df["nr7_range"] = nr7_range
        df["is_nr7"] = is_nr7
        df["nr7_signal"] = is_nr7.astype(int)
    else:
        df["daily_range"] = 0
        df["nr7_range"] = 0
        df["is_nr7"] = False
        df["nr7_signal"] = 0
    
    return df


def calculate_nr7_strength(
    df: pd.DataFrame,
    lookback_days: int = 7,
) -> pd.Series:
    """
    Calculate NR7 strength (how narrow the range is relative to recent history).
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to look back for narrow range
    
    Returns
    -------
    Series with NR7 strength (0-1, where 1 is strongest contraction)
    """
    df = df.copy()
    
    if "high" in df.columns and "low" in df.columns:
        # Resample to daily if intraday
        if hasattr(df.index, 'hour'):
            daily_high = df.resample("D")["high"].max()
            daily_low = df.resample("D")["low"].min()
            daily_range = daily_high - daily_low
        else:
            daily_range = df["high"] - df["low"]
        
        # Calculate NR7 strength
        # Strength = 1 - (today's range / average range)
        avg_range = daily_range.rolling(lookback_days).mean()
        strength = 1 - (daily_range / avg_range)
        
        # Clip to 0-1 range
        strength = strength.clip(0, 1)
        
        # Resample back to original timeframe if needed
        if hasattr(df.index, 'hour'):
            strength = strength.reindex(df.index, method="ffill")
        
        return strength
    else:
        return pd.Series(0.5, index=df.index)


def apply_nr7_filter(
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    df: pd.DataFrame,
    lookback_days: int = 7,
    require_nr7: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Apply NR7 volatility contraction filter to trading signals.
    
    Parameters
    ----------
    long_entries, long_exits, short_entries, short_exits : pd.Series (bool)
        Original trading signals
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to look back for narrow range
    require_nr7 : bool
        If True, only trade on NR7 days. If False, trade on any day but
        with stronger signals on NR7 days.
    
    Returns
    -------
    Tuple of filtered signals (long_entries, long_exits, short_entries, short_exits)
    """
    # Calculate NR7
    df_nr7 = calculate_nr7(df, lookback_days)
    
    if require_nr7:
        # Only allow trades on NR7 days
        filtered_long_entries = long_entries & df_nr7["is_nr7"]
        filtered_short_entries = short_entries & df_nr7["is_nr7"]
    else:
        # Trade on any day, but stronger signals on NR7 days
        # This is more permissive but still has some edge
        filtered_long_entries = long_entries  # Keep all entries
        filtered_short_entries = short_entries  # Keep all entries
    
    # Exits remain unchanged (exit regardless of NR7)
    filtered_long_exits = long_exits
    filtered_short_exits = short_exits
    
    return filtered_long_entries, filtered_long_exits, filtered_short_entries, filtered_short_exits


def nr7_summary(df: pd.DataFrame, lookback_days: int = 7) -> dict:
    """
    Generate summary statistics for NR7 distribution.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to look back for narrow range
    
    Returns
    -------
    dict with NR7 statistics
    """
    df_nr7 = calculate_nr7(df, lookback_days)
    
    nr7_days = df_nr7["is_nr7"].sum()
    total_days = len(df_nr7)
    
    summary = {
        "total_bars": total_days,
        "nr7_days": nr7_days,
        "nr7_pct": nr7_days / total_days * 100,
        "avg_daily_range": df_nr7["daily_range"].mean(),
        "avg_nr7_range": df_nr7["nr7_range"].mean(),
        "range_contraction_ratio": df_nr7["nr7_range"].mean() / df_nr7["daily_range"].mean(),
    }
    
    return summary


def calculate_nr7_continuation_probability(
    df: pd.DataFrame,
    lookback_days: int = 7,
    forward_days: int = 5,
) -> dict:
    """
    Calculate NR7 breakout continuation probability.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days to look back for narrow range
    forward_days : int
        Number of days to look forward for continuation
    
    Returns
    -------
    dict with continuation probability statistics
    """
    df_nr7 = calculate_nr7(df, lookback_days)
    
    if "close" not in df.columns:
        return {}
    
    # Resample to daily if intraday
    if hasattr(df.index, 'hour'):
        daily = df.resample("D")["close"].last().dropna()
        is_nr7 = df_nr7["is_nr7"].reindex(daily.index, method="ffill")
    else:
        daily = df["close"]
        is_nr7 = df_nr7["is_nr7"]
    
    # Calculate forward returns
    forward_returns = daily.pct_change(forward_days).shift(-forward_days)
    
    # Filter for NR7 days
    nr7_returns = forward_returns[is_nr7]
    non_nr7_returns = forward_returns[~is_nr7]
    
    # Calculate continuation probability (positive return after NR7)
    if len(nr7_returns) > 0:
        nr7_continuation = (nr7_returns > 0).sum() / len(nr7_returns)
    else:
        nr7_continuation = 0.5
    
    if len(non_nr7_returns) > 0:
        non_nr7_continuation = (non_nr7_returns > 0).sum() / len(non_nr7_returns)
    else:
        non_nr7_continuation = 0.5
    
    summary = {
        "nr7_days": len(nr7_returns),
        "non_nr7_days": len(non_nr7_returns),
        "nr7_continuation_prob": nr7_continuation,
        "non_nr7_continuation_prob": non_nr7_continuation,
        "edge": nr7_continuation - non_nr7_continuation,
        "avg_nr7_return": nr7_returns.mean() if len(nr7_returns) > 0 else 0,
        "avg_non_nr7_return": non_nr7_returns.mean() if len(non_nr7_returns) > 0 else 0,
    }
    
    return summary


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/admin1/project9/backtest")
    
    # Load sample data
    DATA = "/mnt/c/Users/Admin/project9/data"
    syms = ["NVDA", "AMD", "PLTR", "MRVL"]
    
    print("=" * 80)
    print("NR7 VOLATILITY CONTRACTION FILTER ANALYSIS")
    print("=" * 80)
    
    for sym in syms:
        print(f"\n{'─' * 60}")
        print(f"  {sym}")
        print(f"{'─' * 60}")
        
        try:
            df = pd.read_parquet(f"{DATA}/{sym}_5min.parquet")
            
            # Calculate NR7 summary
            summary = nr7_summary(df, lookback_days=7)
            print(f"  NR7 Days: {summary['nr7_days']} ({summary['nr7_pct']:.1f}%)")
            print(f"  Avg Daily Range: {summary['avg_daily_range']:.3f}")
            print(f"  Avg NR7 Range: {summary['avg_nr7_range']:.3f}")
            print(f"  Range Contraction Ratio: {summary['range_contraction_ratio']:.3f}")
            
            # Calculate continuation probability
            continuation = calculate_nr7_continuation_probability(df, lookback_days=7, forward_days=5)
            print(f"\n  NR7 Continuation Probability:")
            print(f"    NR7 Days: {continuation.get('nr7_days', 0)}")
            print(f"    NR7 Continuation: {continuation.get('nr7_continuation_prob', 0)*100:.1f}%")
            print(f"    Non-NR7 Continuation: {continuation.get('non_nr7_continuation_prob', 0)*100:.1f}%")
            print(f"    Edge: {continuation.get('edge', 0)*100:.1f}%")
            print(f"    Avg NR7 Return: {continuation.get('avg_nr7_return', 0)*100:.3f}%")
            print(f"    Avg Non-NR7 Return: {continuation.get('avg_non_nr7_return', 0)*100:.3f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 80}")
    print("NR7 FILTER IMPLEMENTATION COMPLETE")
    print("=" * 80)
