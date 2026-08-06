"""Volatility-calibrated trailing stops.

Adaptive stop distances based on:
1. Volatility regime (HMM-derived or rolling)
2. Trend strength (directional efficiency ratio)
3. Combined scaling with bounds

Reference: AdaptiveTrend framework (arXiv:2602.11708)
"""

import numpy as np
import pandas as pd


def compute_regime_vol_factor(
    regime_probs: pd.Series,
    vol_scale_range: tuple[float, float] = (0.6, 1.8),
) -> pd.Series:
    """
    Scale factor based on volatility regime probability.

    High P(trending) -> wider stop (let winners run).
    Low P(trending) / mean-reverting -> tighter stop (protect profits).

    Returns values in vol_scale_range.
    """
    low, high = vol_scale_range
    mid = (low + high) / 2
    span = (high - low) / 2

    factor = mid + span * (regime_probs - 0.5) * 2
    return factor.clip(low, high)


def compute_trend_strength_factor(
    df: pd.DataFrame,
    adx_period: int = 14,
    trend_scale_range: tuple[float, float] = (0.8, 1.2),
) -> pd.Series:
    """
    Scale factor based on directional efficiency ratio.

    Strong trend (high |returns| / volatility) -> wider stop.
    Weak/no trend -> tighter stop.

    Returns values in trend_scale_range.
    """
    low, high = trend_scale_range
    mid = (low + high) / 2
    span = (high - low) / 2

    close = df["close"]
    returns = close.pct_change()

    direction = (close - close.shift(adx_period)).abs()
    volatility = returns.rolling(adx_period).std() * np.sqrt(adx_period) * close

    efficiency = direction / (volatility + 1e-10)
    efficiency = efficiency.clip(0, 3)

    normalized = efficiency / 1.5
    factor = mid + span * (normalized - 1)
    return factor.clip(low, high)


def compute_vol_calibrated_stop(
    df: pd.DataFrame,
    atr: pd.Series,
    regime_probs: pd.Series,
    base_mult: float = 3.0,
    vol_scale_range: tuple[float, float] = (0.6, 1.8),
    trend_scale_range: tuple[float, float] = (0.8, 1.2),
) -> pd.Series:
    """
    Dynamic trailing stop distance per bar.

    Logic:
    - High P(trending) -> wider stop (let winners run)
    - Low P(trending) / mean-reverting -> tighter stop (protect profits)
    - Trend strength adjusts further (strong trend = wider)

    Returns price distance for vectorbt sl_trail parameter.
    """
    vol_factor = compute_regime_vol_factor(regime_probs, vol_scale_range)
    trend_factor = compute_trend_strength_factor(df, trend_scale_range=trend_scale_range)

    final_mult = base_mult * vol_factor * trend_factor
    final_mult = final_mult.clip(base_mult * 0.5, base_mult * 2.0)

    return atr * final_mult
