"""Short volatility strategy — SPY.

Sells when realized volatility is low (overpriced vol premium) and buys back
when volatility spikes. Based on the principle that volatility is mean-reverting
and options tend to be overpriced (volatility risk premium).

Strategy:
- Long entry (short vol): when 20-period realized vol < 20th percentile
- Long exit (cover): when vol spikes above 80th percentile
- Short entry: N/A (long-only short vol)
- Short exit: N/A
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    vol_period: int = 20,
    vol_lookback: int = 252,
    entry_percentile: float = 0.25,
    exit_percentile: float = 0.75,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 3.0,
    long_only: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Short volatility strategy: sell vol when it's low, cover when it spikes.

    Returns (long_entries, long_exits, short_entries, short_exits, trailing_stops).

    Long entry (sell vol position): realized vol < entry_percentile of lookback
    Long exit (cover): realized vol > exit_percentile of lookback
    
    Note: 'long' here means 'long the short-vol position' (i.e. we're short vol
    and the position makes money when vol contracts or stays low).
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    realized_vol = log_returns.rolling(vol_period).std() * np.sqrt(252 * 390)

    vol_low = realized_vol.rolling(vol_lookback, min_periods=vol_lookback).quantile(entry_percentile)
    vol_high = realized_vol.rolling(vol_lookback, min_periods=vol_lookback).quantile(exit_percentile)

    # Enter short vol position when vol is low
    long_entries = (realized_vol < vol_low) & vol_low.notna()
    
    # Exit (cover) when vol spikes
    long_exits = (realized_vol > vol_high) & vol_high.notna()

    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    # Trailing stop based on ATR (wider for vol strategies)
    from risk.position_sizer import compute_atr
    atr = compute_atr(df, 14)
    trailing_stops = atr * trail_atr_mult

    return (
        long_entries.shift(1).fillna(False),
        long_exits.shift(1).fillna(False),
        short_entries,
        short_exits,
        trailing_stops.shift(1).fillna(0),
    )
