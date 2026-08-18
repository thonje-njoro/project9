"""
strategies/cper_gld_ratio.py — Copper/Gold ratio pair trade (mean reversion).
Synthetic instrument: ratio = CPER_close / GLD_close.
Applies to: CPER_GLD on 4H timeframe.
"""

import logging

import numpy as np
import pandas as pd

from backtest.data.resampler import compute_atr

logger = logging.getLogger(__name__)


def generate_signals(df: pd.DataFrame,
                     z_entry: float = 1.8,
                     z_exit: float = 0.0,
                     z_take_profit: float = 0.0,
                     window: int = 20,
                     use_trailing_stop: bool = True,
                     trail_atr_mult: float = 2.0,
                     atr_period: int = 14,
                     **kwargs) -> tuple:
    """
    CPER/GLD ratio mean reversion.

    Computes rolling Z-score of the ratio. Enters long CPER / short GLD
    when Z < -z_entry. Exits at Z > -z_exit or Z > z_take_profit.

    Args:
        df: OHLCV DataFrame for the synthetic CPER/GLD ratio.
        z_entry: Z-score threshold for entry.
        z_exit: Z-score threshold for exit.
        z_take_profit: Z-score for take profit.
        window: Rolling window for Z-score computation.
        use_trailing_stop: Enable ATR trailing stop.
        trail_atr_mult: ATR multiplier for trailing stop.
        atr_period: ATR lookback period.

    Returns:
        4-tuple: (long_entries, long_exits, short_entries, short_exits)
    """
    if df.empty:
        empty = pd.Series(dtype=bool)
        return empty, empty, empty, empty

    close = df["close"]
    n = len(df)

    # ─── Compute Z-score of ratio ────────────────────────────────────────────
    ratio_mean = close.rolling(window=window, min_periods=5).mean()
    ratio_std = close.rolling(window=window, min_periods=5).std()
    ratio_std = ratio_std.replace(0, np.nan).bfill().fillna(1.0)
    z_score = (close - ratio_mean) / ratio_std

    # ─── ATR on ratio series ─────────────────────────────────────────────────
    atr = compute_atr(df, period=atr_period)
    atr = atr.bfill().fillna(0)

    # ─── Entry signals ───────────────────────────────────────────────────────
    # Long CPER / Short GLD when ratio is cheap (Z < -z_entry)
    long_entries = (z_score < -z_entry) & (z_score.shift(1) >= -z_entry)

    # Short CPER / Long GLD when ratio is expensive (Z > z_entry)
    short_entries = (z_score > z_entry) & (z_score.shift(1) <= z_entry)

    # ─── Exit signals ────────────────────────────────────────────────────────
    # Exit long when Z reverts to z_exit or take profit
    long_exits = (z_score > -z_exit) & (z_score.shift(1) <= -z_exit)
    if z_take_profit > 0:
        long_exits = long_exits | (z_score > z_take_profit)

    # Exit short when Z reverts
    short_exits = (z_score < z_exit) & (z_score.shift(1) >= z_exit)
    if z_take_profit > 0:
        short_exits = short_exits | (z_score < -z_take_profit)

    # ─── Trailing stop ───────────────────────────────────────────────────────
    if use_trailing_stop:
        trailing_stops = pd.Series(np.nan, index=df.index)
        in_position = False
        position_side = None
        stop_price = 0.0

        for i in range(n):
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

            if long_entries.iloc[i] and not in_position:
                in_position = True
                position_side = "long"
                stop_price = close.iloc[i] - trail_atr_mult * current_atr

            elif short_entries.iloc[i] and not in_position:
                in_position = True
                position_side = "short"
                stop_price = close.iloc[i] + trail_atr_mult * current_atr

            if in_position:
                if position_side == "long":
                    new_stop = close.iloc[i] - trail_atr_mult * current_atr
                    stop_price = max(stop_price, new_stop)
                    if close.iloc[i] <= stop_price:
                        long_exits.iloc[i] = True
                        in_position = False

                elif position_side == "short":
                    new_stop = close.iloc[i] + trail_atr_mult * current_atr
                    stop_price = min(stop_price, new_stop)
                    if close.iloc[i] >= stop_price:
                        short_exits.iloc[i] = True
                        in_position = False

            trailing_stops.iloc[i] = stop_price if in_position else np.nan

    # ─── Anti-lookahead: shift by 1 ──────────────────────────────────────────
    long_entries = long_entries.shift(1).fillna(False).astype(bool)
    long_exits = long_exits.shift(1).fillna(False).astype(bool)
    short_entries = short_entries.shift(1).fillna(False).astype(bool)
    short_exits = short_exits.shift(1).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits
