"""
risk/regime_filter.py — Drift regime gate (UpFraction).
From arxiv "Discovery of a 13-Sharpe OOS Factor" (NASA researcher, Nov 2025).
"""

import logging

import numpy as np
import pandas as pd

from backtest.config import RISK_CONFIG

logger = logging.getLogger(__name__)


def compute_regime_gate(daily_close: pd.Series,
                        lookback: int | None = None,
                        entry_thresh: float | None = None,
                        exit_thresh: float | None = None) -> pd.Series:
    """
    UpFraction = % of positive days in trailing lookback-day window.
    Returns boolean Series: True = regime allows trading.

    Uses hysteresis: enter when up_fraction > entry_thresh, exit < exit_thresh.
    This prevents rapid toggling at the boundary.

    Args:
        daily_close: Daily close price series (DatetimeIndex).
        lookback: Rolling window in days. Default from RISK_CONFIG.
        entry_thresh: Min UpFraction to allow entries. Default from RISK_CONFIG.
        exit_thresh: UpFraction below which to exit. Default from RISK_CONFIG.

    Returns:
        pd.Series[bool]: True where trading is allowed.
    """
    if lookback is None:
        lookback = RISK_CONFIG["regime_lookback_days"]
    if entry_thresh is None:
        entry_thresh = RISK_CONFIG["regime_entry_thresh"]
    if exit_thresh is None:
        exit_thresh = RISK_CONFIG["regime_exit_thresh"]

    daily_returns = daily_close.pct_change()
    up_fraction = (daily_returns > 0).rolling(lookback, min_periods=lookback).mean()

    # Hysteresis state machine
    in_regime = pd.Series(False, index=daily_close.index, dtype=bool)
    state = False

    for i in range(len(up_fraction)):
        uf = up_fraction.iloc[i]
        if pd.isna(uf):
            in_regime.iloc[i] = False
            continue
        if not state and uf > entry_thresh:
            state = True
        elif state and uf < exit_thresh:
            state = False
        in_regime.iloc[i] = state

    return in_regime


def apply_regime_to_intraday(regime_daily: pd.Series,
                             intraday_index: pd.DatetimeIndex) -> pd.Series:
    """
    Map a daily regime gate to an intraday index via forward-fill.

    Args:
        regime_daily: Daily boolean series (date-indexed).
        intraday_index: Intraday DatetimeIndex to map onto.

    Returns:
        pd.Series[bool] aligned to intraday_index.
    """
    # Normalize intraday index to date objects (tz-safe)
    if intraday_index.tz is not None:
        dates_utc = intraday_index.tz_convert("UTC")
    else:
        dates_utc = intraday_index

    daily_dates = pd.Series(
        dates_utc.normalize().date,  # date objects, no tz ambiguity
        index=intraday_index,
    )

    # Build date→value mapping from regime_daily
    regime_by_date = {}
    if isinstance(regime_daily.index[0], pd.Timestamp):
        for ts, val in regime_daily.items():
            regime_by_date[ts.date()] = val
    else:
        for d, val in regime_daily.items():
            regime_by_date[d] = val

    # Map and forward-fill
    mapped = daily_dates.map(regime_by_date).ffill().fillna(False).astype(bool)
    mapped.index = intraday_index
    return mapped


def compute_regime_coverage(regime_gate: pd.Series) -> float:
    """Return the fraction of bars where regime allows trading."""
    if len(regime_gate) == 0:
        return 0.0
    return float(regime_gate.sum() / len(regime_gate))
