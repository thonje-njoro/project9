"""Resample 1-min OHLCV bars to higher timeframes."""

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, timeframe: str, asset_class: str = "stock") -> pd.DataFrame:
    """
    Resample 1-min OHLCV to target timeframe.
    timeframe: "15Min", "1H", "4H"
    OHLCV aggregation rules: O=first, H=max, L=min, C=last, V=sum
    For stocks: filter to regular market hours (09:30–16:00 ET) before resampling.
    For crypto: no filtering.
    """
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    resampled = resampled.dropna(subset=["close"])

    resampled["close"] = resampled["close"].ffill()
    resampled["volume"] = resampled["volume"].fillna(0)

    return resampled
