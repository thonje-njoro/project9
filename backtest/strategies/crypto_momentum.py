"""Crypto momentum strategy — BTC/USD.

Simple trend-following momentum on hourly BTC data with ATR-based
volatility stops. Crypto markets have strong trending behavior and
higher volatility, making momentum approaches effective.

Strategy:
- Long entry: fast EMA > slow EMA (bullish cross)
- Long exit: fast EMA < slow EMA (bearish cross) OR trailing stop hit
- ATR trailing stop to protect profits during volatile pullbacks
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    fast_ema: int = 20,
    slow_ema: int = 50,
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 3.0,
    use_vol_filter: bool = True,
    vol_threshold: float = 0.05,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    BTC momentum with EMA crossover and volatility-aware stops.

    Returns (long_entries, long_exits, short_entries, short_exits, trailing_stops).

    Long entry: fast EMA crosses above slow EMA
    Long exit: fast EMA crosses below slow EMA OR trailing stop activates
    No short entries (crypto has asymmetric upside).
    """
    close = df["close"]
    fast = close.ewm(span=fast_ema, adjust=False).mean()
    slow = close.ewm(span=slow_ema, adjust=False).mean()

    cross_above = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_below = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    # Volatility filter: don't enter during extreme volatility
    log_returns = np.log(close / close.shift(1))
    recent_vol = log_returns.rolling(20).std()
    vol_ok = recent_vol < log_returns.rolling(100).quantile(0.90) if use_vol_filter else pd.Series(True, index=df.index)

    long_entries = cross_above & vol_ok
    long_exits = cross_below

    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

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
