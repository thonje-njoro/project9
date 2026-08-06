"""Gold/Oil ratio mean reversion strategy.

Trades the ratio between GLD (gold ETF) and USO (oil ETF).
When the ratio deviates significantly from its historical mean, bet on reversion.

Gold/oil ratio is a classic macro pair — gold tends to rise during uncertainty,
oil during growth. The ratio mean-reverts over multi-week periods.

Strategy:
- Compute GLD/USO price ratio
- Entry: ratio > mean + threshold * std (short ratio) or ratio < mean - threshold * std (long ratio)
- Exit: ratio crosses back to mean
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    ratio_period: int = 60,
    std_threshold: float = 2.0,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    GLD/USO ratio mean reversion.

    Returns (long_entries, long_exits, short_entries, short_exits, trailing_stops).
    
    Long ratio (long GLD, short USO): when ratio is below lower band
    Short ratio (short GLD, long USO): when ratio is above upper band
    Exit both: when ratio crosses the mean again
    """
    close = df["close"]
    ratio = close

    ratio_mean = ratio.rolling(ratio_period).mean()
    ratio_std = ratio.rolling(ratio_period).std()

    upper = ratio_mean + std_threshold * ratio_std
    lower = ratio_mean - std_threshold * ratio_std

    # Long ratio: ratio is below lower band (gold is cheap vs oil)
    long_entries = ratio < lower
    long_exits = ratio >= ratio_mean

    # Short ratio: ratio is above upper band (gold is expensive vs oil)
    short_entries = ratio > upper
    short_exits = ratio <= ratio_mean

    from risk.position_sizer import compute_atr
    atr = compute_atr(df, 14)
    trailing_stops = atr * trail_atr_mult

    return (
        long_entries.shift(1).fillna(False),
        long_exits.shift(1).fillna(False),
        short_entries.shift(1).fillna(False),
        short_exits.shift(1).fillna(False),
        trailing_stops.shift(1).fillna(0),
    )
