"""Trend following strategy — GLD, USO."""

import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    fast_ema: int = 50,
    slow_ema: int = 200,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Long entry:  fast EMA crosses above slow EMA
    Long exit:   fast EMA crosses below slow EMA
    Short entry: fast EMA crosses below slow EMA
    Short exit:  fast EMA crosses above slow EMA
    """
    fast = df["close"].ewm(span=fast_ema, adjust=False).mean()
    slow = df["close"].ewm(span=slow_ema, adjust=False).mean()

    cross_above = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_below = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    return (
        cross_above.shift(1).fillna(False),
        cross_below.shift(1).fillna(False),
        cross_below.shift(1).fillna(False),
        cross_above.shift(1).fillna(False),
    )
