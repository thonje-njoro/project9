"""
strategies/momentum_orb.py — Momentum Opening Range Breakout strategy.
Extended opening range (60min for NVDA/AMD/PLTR, 5min for MRVL) on 5-min bars.
Applies to: NVDA_MORB, AMD_MORB, PLTR_MORB, MRVL_MORB.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_atr_fallback(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR with fallback for short history.
    If fewer than `period` bars exist, use range/2 as ATR.
    Never returns NaN.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=1).mean()

    # Fallback: if still NaN, use range/2
    price_range = high - low
    atr = atr.fillna(price_range / 2)

    # Final fallback: fill any remaining NaN with 0
    atr = atr.fillna(0)

    # Anti-lookahead shift
    return atr.shift(1).fillna(0)


def generate_signals(df: pd.DataFrame,
                     or_minutes: int = 60,
                     atr_mult_stop: float = 1.5,
                     trail_mult: float = 1.5,
                     min_price: float = 5.0,
                     min_volume: float = 100_000,
                     session_open_hour: int = 9,
                     session_open_minute: int = 30,
                     session_close_hour: int = 16,
                     tz: str = "US/Eastern",
                     atr_period: int = 14,
                     **kwargs) -> tuple:
    """
    Momentum ORB signal generation.

    Design decisions (confirmed from walk-forward validation):
    - NO trend filter (200-day MA degrades OOS performance)
    - NO min_or_range_atr filter (tends to overfit)
    - Trailing stop IS used (locks in momentum profits)
    - All timezone comparisons in US/Eastern

    Args:
        df: OHLCV DataFrame with DatetimeIndex (UTC).
        or_minutes: Opening range duration in minutes.
        atr_mult_stop: ATR multiplier for initial stop.
        trail_mult: ATR multiplier for trailing stop.
        min_price: Minimum price filter.
        min_volume: Minimum bar volume filter.
        session_open_hour: Market open hour in ET.
        session_open_minute: Market open minute in ET.
        session_close_hour: Market close hour in ET.
        tz: Timezone for session comparisons.
        atr_period: ATR lookback period.

    Returns:
        4-tuple: (long_entries, long_exits, short_entries, short_exits)
    """
    if df.empty:
        return (pd.Series(dtype=bool), pd.Series(dtype=bool),
                pd.Series(dtype=bool), pd.Series(dtype=bool))

    df = df.copy()

    # ─── CRITICAL: Timezone conversion to ET ─────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    n = len(df)
    long_entries = pd.Series(False, index=df.index)
    long_exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    # ─── ATR with fallback ───────────────────────────────────────────────────
    atr = _compute_atr_fallback(df, period=atr_period)

    # ─── Session identification ──────────────────────────────────────────────
    dates = df.index.date
    unique_dates = sorted(set(dates))

    session_open = session_open_hour * 60 + session_open_minute
    session_close = session_close_hour * 60

    # Number of 5-min bars in opening range
    or_bars_count = or_minutes // 5

    for date in unique_dates:
        mask = dates == date
        day_bars = df[mask]
        if len(day_bars) < or_bars_count + 2:
            continue

        # ─── Filters ─────────────────────────────────────────────────────────
        if day_bars["close"].mean() < min_price:
            continue

        if day_bars["volume"].sum() < min_volume:
            continue

        # ─── Find opening range bars ─────────────────────────────────────────
        or_mask = (
            (day_bars.index.hour * 60 + day_bars.index.minute >= session_open) &
            (day_bars.index.hour * 60 + day_bars.index.minute < session_open + or_minutes)
        )
        or_bars = day_bars[or_mask]

        if len(or_bars) < 1:
            continue

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()

        # ─── Post-OR bars ────────────────────────────────────────────────────
        post_or_mask = (
            (day_bars.index.hour * 60 + day_bars.index.minute >= session_open + or_minutes) &
            (day_bars.index.hour * 60 + day_bars.index.minute < session_close)
        )
        post_or_bars = day_bars[post_or_mask]

        if post_or_bars.empty:
            continue

        # ─── Entry + trailing stop logic ─────────────────────────────────────
        in_position = False
        position_side = None
        entry_price = 0.0
        trail_stop = 0.0
        highest_since_entry = 0.0
        lowest_since_entry = np.inf

        for j in range(len(post_or_bars)):
            bar = post_or_bars.iloc[j]
            bar_idx = post_or_bars.index[j]
            current_atr = atr.loc[bar_idx] if bar_idx in atr.index else 0

            if not in_position:
                # Long: close above OR high
                if bar["close"] > or_high:
                    # Entry on next bar
                    next_j = j + 1
                    if next_j < len(post_or_bars):
                        entry_idx = post_or_bars.index[next_j]
                        entry_price = post_or_bars.iloc[next_j]["open"]
                        long_entries.loc[entry_idx] = True
                        in_position = True
                        position_side = "long"
                        highest_since_entry = entry_price
                        trail_stop = entry_price - atr_mult_stop * current_atr

                # Short: close below OR low
                elif bar["close"] < or_low:
                    next_j = j + 1
                    if next_j < len(post_or_bars):
                        entry_idx = post_or_bars.index[next_j]
                        entry_price = post_or_bars.iloc[next_j]["open"]
                        short_entries.loc[entry_idx] = True
                        in_position = True
                        position_side = "short"
                        lowest_since_entry = entry_price
                        trail_stop = entry_price + atr_mult_stop * current_atr

            elif in_position:
                if position_side == "long":
                    highest_since_entry = max(highest_since_entry, bar["high"])
                    # Trailing stop: move up only
                    new_trail = highest_since_entry - trail_mult * current_atr
                    trail_stop = max(trail_stop, new_trail)

                    if bar["low"] <= trail_stop:
                        long_exits.loc[bar_idx] = True
                        in_position = False

                elif position_side == "short":
                    lowest_since_entry = min(lowest_since_entry, bar["low"])
                    new_trail = lowest_since_entry + trail_mult * current_atr
                    trail_stop = min(trail_stop, new_trail)

                    if bar["high"] >= trail_stop:
                        short_exits.loc[bar_idx] = True
                        in_position = False

        # ─── Force close at session end ──────────────────────────────────────
        session_end_mask = (
            (day_bars.index.hour * 60 + day_bars.index.minute >= session_close - 5) &
            (day_bars.index.hour * 60 + day_bars.index.minute < session_close)
        )
        session_end_bars = day_bars[session_end_mask]
        if not session_end_bars.empty and in_position:
            last_idx = session_end_bars.index[-1]
            if position_side == "long":
                long_exits.loc[last_idx] = True
            elif position_side == "short":
                short_exits.loc[last_idx] = True

    # ─── Anti-lookahead: shift by 1 ──────────────────────────────────────────
    long_entries = long_entries.shift(1).fillna(False).astype(bool)
    long_exits = long_exits.shift(1).fillna(False).astype(bool)
    short_entries = short_entries.shift(1).fillna(False).astype(bool)
    short_exits = short_exits.shift(1).fillna(False).astype(bool)

    # Convert back to UTC
    df.index = df.index.tz_convert("UTC")
    long_entries.index = df.index
    long_exits.index = df.index
    short_entries.index = df.index
    short_exits.index = df.index

    return long_entries, long_exits, short_entries, short_exits
