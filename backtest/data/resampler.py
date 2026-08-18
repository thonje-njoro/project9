"""
data/resampler.py — OHLCV resampling with timezone awareness.
Resamples 1Min bars to 5Min/15Min/1H/4H.
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def resample_ohlcv(df: pd.DataFrame, target_tf: str,
                   tz: str | None = None) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a higher timeframe.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and DatetimeIndex.
        target_tf: Target timeframe string: '5min', '15min', '1h', '4h', '1d'.
        tz: If set, convert index to this timezone before resampling.

    Returns:
        Resampled DataFrame with same column structure.
    """
    if df.empty:
        return df

    # Make a copy to avoid mutating original
    df = df.copy()

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Timezone conversion
    if tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(tz)

    # Map timeframe strings to pandas offset aliases
    tf_map = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    pd_tf = tf_map.get(target_tf)
    if pd_tf is None:
        raise ValueError(f"Unknown target timeframe: {target_tf}")

    # OHLCV resampling rules
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    resampled = df.resample(pd_tf).agg(agg).dropna(subset=["open"])

    # Restore to UTC if we converted
    if tz is not None and resampled.index.tz is not None:
        resampled.index = resampled.index.tz_convert("UTC")

    return resampled


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: resample to daily bars."""
    return resample_ohlcv(df, "1d")


def compute_vwap(df: pd.DataFrame, reset_daily: bool = True) -> pd.Series:
    """
    Compute Volume-Weighted Average Price.

    Args:
        df: DataFrame with high, low, close, volume columns.
        reset_daily: If True, VWAP resets at each calendar day boundary.

    Returns:
        pd.Series of VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

    if reset_daily:
        # Group by date and compute cumulative within each day
        dates = df.index.date
        cum_pv = (typical_price * df["volume"]).groupby(dates).cumsum()
        cum_vol = df["volume"].groupby(dates).cumsum()
    else:
        cum_pv = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()

    vwap = cum_pv / cum_vol
    vwap = vwap.replace([np.inf, -np.inf], np.nan)
    return vwap


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range.

    Uses shift(1) to prevent lookahead bias — ATR at bar t uses bars up to t-1.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr.shift(1)  # Anti-lookahead: ATR available at bar t is computed from bars before t
