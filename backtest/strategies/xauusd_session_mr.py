"""
strategies/xauusd_session_mr.py — London+NY session mean reversion on XAUUSD.
1-hour bars fetched from LSE API.
"""

import logging

import numpy as np
import pandas as pd

from backtest.data.resampler import compute_atr, compute_vwap

logger = logging.getLogger(__name__)

# Known high-impact event dates (NFP, FOMC, CPI) — 2022-2024
# In production, load from a calendar file. This is a starter set.
RED_FOLDER_DATES = [
    # 2022
    "2022-01-05", "2022-01-12", "2022-01-26", "2022-02-02", "2022-02-10",
    "2022-03-02", "2022-03-09", "2022-03-16", "2022-04-06", "2022-04-13",
    "2022-05-04", "2022-05-11", "2022-06-01", "2022-06-10", "2022-06-15",
    "2022-07-06", "2022-07-13", "2022-07-27", "2022-08-03", "2022-08-10",
    "2022-09-07", "2022-09-13", "2022-09-21", "2022-10-05", "2022-10-13",
    "2022-11-02", "2022-11-10", "2022-12-07", "2022-12-14",
    # 2023
    "2023-01-04", "2023-01-12", "2023-02-01", "2023-02-14", "2023-03-08",
    "2023-03-14", "2023-03-22", "2023-04-05", "2023-04-12", "2023-05-03",
    "2023-05-10", "2023-06-07", "2023-06-13", "2023-06-14", "2023-07-05",
    "2023-07-12", "2023-07-26", "2023-08-02", "2023-08-10", "2023-09-06",
    "2023-09-13", "2023-09-20", "2023-10-04", "2023-10-12", "2023-11-01",
    "2023-11-14", "2023-12-06", "2023-12-13",
    # 2024
    "2024-01-04", "2024-01-11", "2024-01-31", "2024-02-13", "2024-03-06",
    "2024-03-12", "2024-03-20", "2024-04-03", "2024-04-10", "2024-05-01",
    "2024-05-15", "2024-06-05", "2024-06-12", "2024-07-03", "2024-07-11",
    "2024-07-31", "2024-08-07", "2024-09-04", "2024-09-11", "2024-09-18",
    "2024-10-02", "2024-10-10", "2024-11-06", "2024-11-13", "2024-12-04",
    "2024-12-11", "2024-12-18",
]


def generate_signals(df: pd.DataFrame,
                     z_entry: float = 1.5,
                     z_exit: float = 0.0,
                     london_start_utc: int = 8,
                     london_end_utc: int = 12,
                     ny_start_utc: int = 13,
                     ny_start_minute_utc: int = 30,
                     ny_end_utc: int = 17,
                     spread_pips: float = 1.5,
                     pip_value: float = 0.01,
                     asian_range_multiplier: float = 2.0,
                     atr_period: int = 14,
                     trail_atr_mult: float = 2.0,
                     use_trailing_stop: bool = True,
                     **kwargs) -> tuple:
    """
    XAUUSD session mean reversion.

    London session establishes reference (VWAP).
    NY session mean-reverts to London VWAP.

    Pitfall mitigations:
    - NFP/FOMC/CPI dates: skip entirely
    - Spread modeled: 1.5 pip bid-ask cost
    - Asian session range filter: skip if range > 2× average

    Args:
        df: OHLCV DataFrame with DatetimeIndex (UTC).
        z_entry: ATR multiple for entry threshold.
        z_exit: ATR multiple for exit (0 = revert to VWAP).
        london_start_utc/end_utc: London session hours (UTC).
        ny_start_utc/ny_start_minute_utc: NY session start (UTC).
        ny_end_utc: NY session end (UTC).
        spread_pips: Bid-ask spread in pips.
        pip_value: Value of 1 pip.
        asian_range_multiplier: Skip if Asian range > this × average.
        atr_period: ATR lookback.
        trail_atr_mult: Trailing stop ATR multiplier.
        use_trailing_stop: Enable trailing stop.

    Returns:
        4-tuple: (long_entries, long_exits, short_entries, short_exits)
    """
    if df.empty:
        empty = pd.Series(dtype=bool)
        return empty, empty, empty, empty

    df = df.copy()

    # Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    n = len(df)
    long_entries = pd.Series(False, index=df.index)
    long_exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    # ─── ATR ─────────────────────────────────────────────────────────────────
    atr = compute_atr(df, period=atr_period)
    atr = atr.bfill().fillna(0)

    # ─── Spread cost ─────────────────────────────────────────────────────────
    spread_cost = spread_pips * pip_value

    # ─── Red folder filter ───────────────────────────────────────────────────
    red_dates = set(pd.to_datetime(RED_FOLDER_DATES).date)

    # ─── Daily data for Asian range filter ───────────────────────────────────
    dates = df.index.date
    unique_dates = sorted(set(dates))

    # Asian session: 00:00-08:00 UTC
    # Compute daily Asian range
    asian_ranges = {}
    for date in unique_dates:
        day_mask = dates == date
        day_bars = df[day_mask]
        asian_mask = day_bars.index.hour < london_start_utc
        asian_bars = day_bars[asian_mask]
        if not asian_bars.empty:
            asian_ranges[date] = asian_bars["high"].max() - asian_bars["low"].min()

    avg_asian_range = np.mean(list(asian_ranges.values())) if asian_ranges else 0

    for date in unique_dates:
        # Skip red folder days
        if date in red_dates:
            continue

        day_mask = dates == date
        day_bars = df[day_mask]

        # ─── Asian range filter ──────────────────────────────────────────────
        asian_range = asian_ranges.get(date, 0)
        if avg_asian_range > 0 and asian_range > asian_range_multiplier * avg_asian_range:
            continue

        # ─── London session: compute VWAP ────────────────────────────────────
        london_mask = (
            (day_bars.index.hour >= london_start_utc) &
            (day_bars.index.hour < london_end_utc)
        )
        london_bars = day_bars[london_mask]

        if london_bars.empty:
            continue

        # London VWAP
        tp = (london_bars["high"] + london_bars["low"] + london_bars["close"]) / 3
        cum_pv = (tp * london_bars["volume"]).cumsum()
        cum_vol = london_bars["volume"].cumsum()
        london_vwap = cum_pv / cum_vol

        if london_vwap.empty or pd.isna(london_vwap.iloc[-1]):
            continue

        london_vwap_ref = london_vwap.iloc[-1]

        # ─── NY session: entry signals ───────────────────────────────────────
        ny_mask = (
            (day_bars.index.hour * 60 + day_bars.index.minute >= ny_start_utc * 60 + ny_start_minute_utc) &
            (day_bars.index.hour * 60 + day_bars.index.minute < ny_end_utc * 60)
        )
        ny_bars = day_bars[ny_mask]

        if ny_bars.empty:
            continue

        # Use London ATR for threshold
        london_atr = atr.loc[london_bars.index].mean() if not london_bars.empty else atr.mean()
        if london_atr <= 0:
            london_atr = atr.mean()

        threshold = z_entry * london_atr + spread_cost

        # Check first NY bar for deviation
        first_ny_price = ny_bars["close"].iloc[0]
        deviation = first_ny_price - london_vwap_ref

        if deviation > threshold:
            # Price too high vs London VWAP → short
            if len(ny_bars) > 1:
                entry_idx = ny_bars.index[1]
                short_entries.loc[entry_idx] = True

                # Exit at VWAP reversion or NY close
                for k in range(1, len(ny_bars)):
                    bar_idx = ny_bars.index[k]
                    if ny_bars["close"].iloc[k] <= london_vwap_ref:
                        short_exits.loc[bar_idx] = True
                        break

        elif deviation < -threshold:
            # Price too low vs London VWAP → long
            if len(ny_bars) > 1:
                entry_idx = ny_bars.index[1]
                long_entries.loc[entry_idx] = True

                for k in range(1, len(ny_bars)):
                    bar_idx = ny_bars.index[k]
                    if ny_bars["close"].iloc[k] >= london_vwap_ref:
                        long_exits.loc[bar_idx] = True
                        break

        # ─── Force close at NY end ───────────────────────────────────────────
        last_ny_idx = ny_bars.index[-1]
        if not long_exits.loc[last_ny_idx]:
            # If still in position, force close
            if long_entries.any():
                long_exits.loc[last_ny_idx] = True
        if not short_exits.loc[last_ny_idx]:
            if short_entries.any():
                short_exits.loc[last_ny_idx] = True

    # ─── Anti-lookahead: shift by 1 ──────────────────────────────────────────
    long_entries = long_entries.shift(1).fillna(False).astype(bool)
    long_exits = long_exits.shift(1).fillna(False).astype(bool)
    short_entries = short_entries.shift(1).fillna(False).astype(bool)
    short_exits = short_exits.shift(1).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits
