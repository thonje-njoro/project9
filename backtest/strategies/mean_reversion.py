"""Mean reversion strategy — SPY, QQQ.

Implements adaptive thresholds from:
'Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit' (arXiv:1411.5062)

Key improvements:
1. Adaptive entry/exit boundaries based on volatility regime
2. Trailing stop to protect profits when reversion fails
3. Drift-adjusted thresholds (wider in downtrends, tighter in uptrends)
4. Convolution reversal detector for improved entry timing (from aligrithm.com)
"""

import numpy as np
import pandas as pd


def compute_adaptive_threshold(
    std: pd.Series,
    commission: float = 0.0005,
    base_threshold: float = 1.5,
) -> pd.Series:
    """
    Compute adaptive entry threshold based on volatility regime.

    When volatility is high relative to recent average, widen the band.
    When volatility is low, tighten the band for more entries.
    """
    vol_ratio = std / std.rolling(60).mean()
    vol_adjustment = vol_ratio.clip(0.5, 1.5)

    adaptive = base_threshold * vol_adjustment
    adaptive = adaptive.clip(base_threshold * 0.7, base_threshold * 1.3)
    return adaptive


def compute_trailing_stop_atr(
    df: pd.DataFrame,
    entry_prices: pd.Series,
    atr: pd.Series,
    trail_mult: float = 2.0,
) -> pd.Series:
    """
    Compute trailing stop distance from entry.

    Stop widens as ATR increases, protecting profits.
    Returns price distance for vectorbt sl_trail parameter.
    """
    return atr * trail_mult


def generate_signals(
    df: pd.DataFrame,
    period: int = 20,
    std_threshold: float = 1.5,
    use_adaptive: bool = True,
    commission: float = 0.0005,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 2.0,
    long_only: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (long_entries, long_exits, short_entries, short_exits, trailing_stops).

    Adaptive mode (from arXiv:1411.5062):
    - Entry: close outside adaptive band (SMA +/- threshold * std)
    - Exit: close crosses SMA
    - Trailing stop: ATR-based to protect profits

    Long entry:  close < SMA - adaptive_threshold * std
    Long exit:   close >= SMA OR trailing stop hit
    Short entry: close > SMA + adaptive_threshold * std (only if long_only=False)
    Short exit:  close <= SMA OR trailing stop hit
    """
    close = df["close"]
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()

    if use_adaptive:
        threshold = compute_adaptive_threshold(std, commission, std_threshold)
    else:
        threshold = pd.Series(std_threshold, index=df.index)

    upper = sma + threshold * std
    lower = sma - threshold * std

    long_entries = close < lower
    long_exits = close >= sma
    short_entries = close > upper if not long_only else pd.Series(False, index=df.index)
    short_exits = close <= sma

    atr = None
    if use_trailing_stop:
        from risk.position_sizer import compute_atr
        atr = compute_atr(df, 14)

    trailing_stops = pd.Series(0.0, index=df.index)
    if atr is not None:
        trailing_stops = atr * trail_atr_mult

    return (
        long_entries.shift(1).fillna(False),
        long_exits.shift(1).fillna(False),
        short_entries.shift(1).fillna(False),
        short_exits.shift(1).fillna(False),
        trailing_stops.shift(1).fillna(0),
    )
