"""
strategies/orb_strategy.py — Opening Range Breakout (ORB) strategy.
Implementation follows Zarattini et al. (2023).
Applies to: SPY_ORB, QQQ_ORB on 15Min timeframe.
"""

import logging

import numpy as np
import pandas as pd

from backtest.data.resampler import compute_atr

logger = logging.getLogger(__name__)


def generate_signals(df: pd.DataFrame,
                     orb_period: int = 1,
                     session_open_hour: int = 9,
                     session_open_minute: int = 30,
                     session_close_hour: int = 16,
                     tz: str = "US/Eastern",
                     rel_vol_lookback: int = 14,
                     min_rel_volume: float = 1.0,
                     atr_period: int = 14,
                     atr_stop_pct: float = 0.10,
                     risk_per_trade: float = 0.01,
                     min_price: float = 5.0,
                     min_avg_volume: float = 1_000_000,
                     **kwargs) -> tuple:
    """
    Opening Range Breakout signal generation.

    CRITICAL: All timezone comparisons in US/Eastern.
    Alpaca IEX returns UTC — convert before comparing session hours.

    Args:
        df: OHLCV DataFrame with DatetimeIndex (UTC).
        orb_period: Number of 15-min bars for opening range.
        session_open_hour: Market open hour in ET.
        session_open_minute: Market open minute in ET.
        session_close_hour: Market close hour in ET.
        tz: Timezone for session hour comparisons.
        rel_vol_lookback: Days for relative volume calculation.
        min_rel_volume: Minimum relative volume at OR time.
        atr_period: ATR lookback period.
        atr_stop_pct: ATR multiplier for initial stop.
        risk_per_trade: Risk per trade as fraction of equity.
        min_price: Minimum price filter.
        min_avg_volume: Minimum average daily volume.

    Returns:
        4-tuple: (long_entries, long_exits, short_entries, short_exits)
    """
    if df.empty:
        return (pd.Series(dtype=bool), pd.Series(dtype=bool),
                pd.Series(dtype=bool), pd.Series(dtype=bool))

    df = df.copy()

    # ─── CRITICAL: Timezone conversion ───────────────────────────────────────
    # Index is UTC from Alpaca. Convert to ET for session hour comparisons.
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

    # ─── Precompute ATR (anti-lookahead: from previous days) ─────────────────
    atr = compute_atr(df, period=atr_period)
    atr = atr.bfill().fillna(0)

    # ─── Session identification ──────────────────────────────────────────────
    dates = df.index.date
    unique_dates = sorted(set(dates))

    session_open = session_open_hour * 60 + session_open_minute
    session_close = session_close_hour * 60

    # ─── Daily average volume for filters ────────────────────────────────────
    daily_vol = volume.groupby(dates).sum()
    avg_daily_vol = daily_vol.rolling(window=rel_vol_lookback, min_periods=1).mean()

    for date in unique_dates:
        # Get bars for this day
        mask = dates == date
        day_bars = df[mask]
        if len(day_bars) < orb_period + 1:
            continue

        # Filter: minimum price
        if day_bars["close"].mean() < min_price:
            continue

        # Filter: minimum average daily volume
        date_key = pd.Timestamp(date)
        if date_key in avg_daily_vol.index:
            if avg_daily_vol.loc[date_key] < min_avg_volume:
                continue

        # ─── Find opening range bars ─────────────────────────────────────────
        # Session open bars: first `orb_period` bars after 9:30 ET
        or_mask = (
            (day_bars.index.hour * 60 + day_bars.index.minute >= session_open) &
            (day_bars.index.hour * 60 + day_bars.index.minute < session_open + orb_period * 15)
        )
        or_bars = day_bars[or_mask]

        if len(or_bars) < orb_period:
            continue

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()

        # ─── Relative volume filter ──────────────────────────────────────────
        or_volume = or_bars["volume"].sum()
        # Average volume at OR time from previous days
        or_time_mask_all = (
            (df.index.hour * 60 + df.index.minute >= session_open) &
            (df.index.hour * 60 + df.index.minute < session_open + orb_period * 15)
        )
        or_vol_by_day = volume[or_time_mask_all].groupby(df[or_time_mask_all].index.date).sum()
        if len(or_vol_by_day) > rel_vol_lookback:
            avg_or_vol = or_vol_by_day.iloc[-rel_vol_lookback-1:-1].mean()
        else:
            avg_or_vol = or_vol_by_day.mean()

        if avg_or_vol > 0:
            rel_vol = or_volume / avg_or_vol
            if rel_vol < min_rel_volume:
                continue

        # ─── Session bars after OR ───────────────────────────────────────────
        post_or_mask = (
            (day_bars.index.hour * 60 + day_bars.index.minute >= session_open + orb_period * 15) &
            (day_bars.index.hour * 60 + day_bars.index.minute < session_close)
        )
        post_or_bars = day_bars[post_or_mask]

        if post_or_bars.empty:
            continue

        # ─── Entry logic ─────────────────────────────────────────────────────
        # Entry on NEXT bar's open after OR high/low is crossed by CLOSE
        in_position = False
        position_side = None

        for j in range(len(post_or_bars)):
            bar = post_or_bars.iloc[j]
            bar_idx = post_or_bars.index[j]

            if not in_position:
                # Long entry: close crosses above OR high
                if bar["close"] > or_high:
                    # Entry on NEXT bar open
                    next_idx = j + 1
                    if next_idx < len(post_or_bars):
                        entry_idx = post_or_bars.index[next_idx]
                        long_entries.loc[entry_idx] = True
                        in_position = True
                        position_side = "long"

                # Short entry: close crosses below OR low
                elif bar["close"] < or_low:
                    next_idx = j + 1
                    if next_idx < len(post_or_bars):
                        entry_idx = post_or_bars.index[next_idx]
                        short_entries.loc[entry_idx] = True
                        in_position = True
                        position_side = "short"

            elif in_position:
                # Check stop loss
                if position_side == "long":
                    stop_level = bar["close"] - atr.loc[bar_idx] * atr_stop_pct if atr.loc[bar_idx] > 0 else 0
                    if bar["low"] <= stop_level:
                        long_exits.loc[bar_idx] = True
                        in_position = False

                elif position_side == "short":
                    stop_level = bar["close"] + atr.loc[bar_idx] * atr_stop_pct if atr.loc[bar_idx] > 0 else 0
                    if bar["high"] >= stop_level:
                        short_exits.loc[bar_idx] = True
                        in_position = False

        # ─── Force close at session end ──────────────────────────────────────
        session_end_mask = day_bars.index.hour * 60 + day_bars.index.minute >= session_close - 15
        session_end_bars = day_bars[session_end_mask]
        if not session_end_bars.empty and in_position:
            last_bar_idx = session_end_bars.index[-1]
            if position_side == "long":
                long_exits.loc[last_bar_idx] = True
            elif position_side == "short":
                short_exits.loc[last_bar_idx] = True

    # ─── Anti-lookahead: shift signals by 1 bar ─────────────────────────────
    long_entries = long_entries.shift(1).fillna(False).astype(bool)
    long_exits = long_exits.shift(1).fillna(False).astype(bool)
    short_entries = short_entries.shift(1).fillna(False).astype(bool)
    short_exits = short_exits.shift(1).fillna(False).astype(bool)

    # Convert index back to UTC
    df.index = df.index.tz_convert("UTC")
    long_entries.index = df.index
    long_exits.index = df.index
    short_entries.index = df.index
    short_exits.index = df.index

    return long_entries, long_exits, short_entries, short_exits
