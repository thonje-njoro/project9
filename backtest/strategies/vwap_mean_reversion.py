"""
strategies/vwap_mean_reversion.py — VWAP Z-score intraday mean reversion.
Applies to: SPY, IWM_VWAP on 15Min timeframe.
"""

import logging

import numpy as np
import pandas as pd

from backtest.data.resampler import compute_atr, compute_vwap

logger = logging.getLogger(__name__)


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX) for trend day filter."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=1).mean()
    atr = atr.replace(0, np.nan)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(window=period, min_periods=1).mean()

    return adx.fillna(0)


def generate_signals(df: pd.DataFrame,
                     z_score_lookback: int = 20,
                     z_entry: float = 2.0,
                     z_exit: float = 0.0,
                     require_reversal: bool = True,
                     reversal_lookback: int = 3,
                     skip_trend_days: bool = True,
                     adx_threshold: float = 25.0,
                     use_relative_volume: bool = True,
                     max_relative_volume: float = 1.5,
                     use_atr_change_filter: bool = True,
                     max_atr_change: float = 0.05,
                     use_time_filter: bool = True,
                     entry_start_hour_et: int = 10,
                     entry_end_hour_et: int = 15,
                     long_only: bool = True,
                     max_hold_bars: int = 48,
                     atr_period: int = 14,
                     trail_atr_mult: float = 2.0,
                     **kwargs) -> tuple:
    """
    VWAP mean reversion signal generation.

    VWAP resets at market open each day (no cross-day pollution).

    Args:
        df: OHLCV DataFrame with DatetimeIndex (UTC).
        z_score_lookback: Bars for Z-score mean/std computation.
        z_entry: Z-score threshold for entry.
        z_exit: Z-score threshold for exit (0 = mean reversion).
        require_reversal: Require reversal confirmation.
        reversal_lookback: Bars to check for reversal.
        skip_trend_days: Skip trading on high-ADX days.
        adx_threshold: ADX above which to skip.
        use_relative_volume: Skip extreme volume bars.
        max_relative_volume: Max relative volume for entry.
        use_atr_change_filter: Skip on ATR spikes.
        max_atr_change: Max 1-bar ATR change fraction.
        use_time_filter: Restrict entry hours.
        entry_start_hour_et: Entry window start (ET).
        entry_end_hour_et: Entry window end (ET).
        long_only: Only take long MR trades.
        max_hold_bars: Max bars to hold position.
        atr_period: ATR lookback.
        trail_atr_mult: ATR multiplier for trailing stop.

    Returns:
        5-tuple: (long_entries, long_exits, short_entries, short_exits, trailing_stops)
    """
    if df.empty:
        empty = pd.Series(dtype=bool)
        return empty, empty, empty, empty, pd.Series(dtype=float)

    df = df.copy()

    # Convert to ET for time filters
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_et = df.index.tz_convert("US/Eastern")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    n = len(df)
    long_entries = pd.Series(False, index=df.index)
    long_exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    # ─── Compute VWAP (resets daily) ─────────────────────────────────────────
    vwap = compute_vwap(df, reset_daily=True)

    # ─── Compute ATR ─────────────────────────────────────────────────────────
    atr = compute_atr(df, period=atr_period)
    atr = atr.bfill().fillna(0)

    # ─── Z-score: (close - vwap_mean) / vwap_std ─────────────────────────────
    vwap_deviation = close - vwap
    vwap_mean = vwap_deviation.rolling(window=z_score_lookback, min_periods=5).mean()
    vwap_std = vwap_deviation.rolling(window=z_score_lookback, min_periods=5).std()
    vwap_std = vwap_std.replace(0, np.nan).bfill().fillna(1.0)
    z_score = (vwap_deviation - vwap_mean) / vwap_std

    # ─── Filters ─────────────────────────────────────────────────────────────
    # ADX trend day filter
    adx = _compute_adx(df) if skip_trend_days else pd.Series(0, index=df.index)

    # Relative volume filter
    avg_volume = volume.rolling(window=20, min_periods=1).mean()
    avg_volume = avg_volume.replace(0, np.nan).bfill().fillna(1)
    rel_volume = volume / avg_volume

    # ATR change filter
    atr_change = atr.pct_change().abs().fillna(0)

    # Time filter
    hours_et = df_et.hour
    minutes_et = df_et.minute
    time_ok = pd.Series(True, index=df.index)
    if use_time_filter:
        time_minutes = hours_et * 60 + minutes_et
        start_min = entry_start_hour_et * 60
        end_min = entry_end_hour_et * 60
        time_ok = (time_minutes >= start_min) & (time_minutes <= end_min)

    # ─── Combined filter ─────────────────────────────────────────────────────
    filter_mask = pd.Series(True, index=df.index)
    if skip_trend_days:
        filter_mask = filter_mask & (adx < adx_threshold)
    if use_relative_volume:
        filter_mask = filter_mask & (rel_volume < max_relative_volume)
    if use_atr_change_filter:
        filter_mask = filter_mask & (atr_change < max_atr_change)
    if use_time_filter:
        filter_mask = filter_mask & time_ok

    # ─── Entry signals ───────────────────────────────────────────────────────
    # Long entry: Z < -z_entry with reversal confirmation
    z_cross_down = (z_score < -z_entry) & (z_score.shift(1) >= -z_entry)

    if require_reversal:
        # Require close > prior bar's close within lookback
        reversal_up = close > close.shift(1)
        reversal_confirmed = reversal_up.rolling(window=reversal_lookback, min_periods=1).max().astype(bool)
        long_entry_signal = z_cross_down & reversal_confirmed
    else:
        long_entry_signal = z_cross_down

    long_entries = long_entry_signal & filter_mask

    # Short entry (if not long_only)
    if not long_only:
        z_cross_up = (z_score > z_entry) & (z_score.shift(1) <= z_entry)
        if require_reversal:
            reversal_down = close < close.shift(1)
            reversal_confirmed_short = reversal_down.rolling(window=reversal_lookback, min_periods=1).max().astype(bool)
            short_entry_signal = z_cross_up & reversal_confirmed_short
        else:
            short_entry_signal = z_cross_up
        short_entries = short_entry_signal & filter_mask

    # ─── Exit signals ────────────────────────────────────────────────────────
    # Exit: Z crosses 0 (mean reversion complete)
    z_revert_long = (z_score > z_exit) & (z_score.shift(1) <= z_exit)
    z_revert_short = (z_score < -z_exit) & (z_score.shift(1) >= -z_exit)

    long_exits = z_revert_long
    short_exits = z_revert_short

    # ─── Trailing stop ───────────────────────────────────────────────────────
    trailing_stops = pd.Series(np.nan, index=df.index)
    in_position = False
    position_side = None
    stop_price = 0.0
    bars_held = 0

    for i in range(n):
        if long_entries.iloc[i] and not in_position:
            in_position = True
            position_side = "long"
            stop_price = close.iloc[i] - trail_atr_mult * atr.iloc[i]
            bars_held = 0
        elif not long_only and short_entries.iloc[i] and not in_position:
            in_position = True
            position_side = "short"
            stop_price = close.iloc[i] + trail_atr_mult * atr.iloc[i]
            bars_held = 0

        if in_position:
            bars_held += 1

            if position_side == "long":
                new_stop = close.iloc[i] - trail_atr_mult * atr.iloc[i]
                stop_price = max(stop_price, new_stop)
                trailing_stops.iloc[i] = stop_price

                if close.iloc[i] <= stop_price or bars_held >= max_hold_bars:
                    long_exits.iloc[i] = True
                    in_position = False

            elif position_side == "short":
                new_stop = close.iloc[i] + trail_atr_mult * atr.iloc[i]
                stop_price = min(stop_price, new_stop)
                trailing_stops.iloc[i] = stop_price

                if close.iloc[i] >= stop_price or bars_held >= max_hold_bars:
                    short_exits.iloc[i] = True
                    in_position = False

    # ─── Anti-lookahead: shift by 1 ──────────────────────────────────────────
    long_entries = long_entries.shift(1).fillna(False).astype(bool)
    long_exits = long_exits.shift(1).fillna(False).astype(bool)
    short_entries = short_entries.shift(1).fillna(False).astype(bool)
    short_exits = short_exits.shift(1).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits, trailing_stops
