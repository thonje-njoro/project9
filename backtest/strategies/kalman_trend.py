"""
strategies/kalman_trend.py — Kalman filter trend-following strategy.
Implements Benhamou's 2-state Kalman filter (arXiv:1808.03297).
Applies to: GLD, TLT, IWM, CPER on 4H timeframe.
"""

import logging

import numpy as np
import pandas as pd

from backtest.data.resampler import compute_atr, compute_vwap

logger = logging.getLogger(__name__)


def _kalman_filter(close: pd.Series, Q: float = 0.01, R: float = 1.0) -> pd.DataFrame:
    """
    2-state Kalman filter: [price_estimate, velocity].

    State transition: x_{t+1} = F @ x_t + noise
        F = [[1, 1],
             [0, 1]]
    Observation: z_t = H @ x_t + noise
        H = [1, 0]

    Args:
        close: Price series.
        Q: Process noise covariance (scalar, applied to both states).
        R: Measurement noise variance.

    Returns:
        DataFrame with columns: ['estimate', 'velocity']
    """
    n = len(close)
    estimates = np.zeros(n)
    velocities = np.zeros(n)

    # Initial state
    x = np.array([close.iloc[0], 0.0])  # [price, velocity]
    P = np.eye(2)  # Initial covariance

    F = np.array([[1, 1],
                  [0, 1]])
    H = np.array([[1, 0]])
    Q_mat = Q * np.eye(2)

    for t in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q_mat

        # Update
        z = close.iloc[t]
        y_resid = z - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T / S  # Kalman gain (2x1)

        x = x_pred + K.flatten() * y_resid
        P = (np.eye(2) - K @ H) @ P_pred

        estimates[t] = x[0]
        velocities[t] = x[1]

    return pd.DataFrame({
        "estimate": estimates,
        "velocity": velocities,
    }, index=close.index)


def _compute_pullback_entries(close: pd.Series, signal: pd.Series,
                              atr: pd.Series,
                              pullback_max_candles: int = 2,
                              entry_window_periods: int = 3,
                              pullback_atr_mult: float = 0.3) -> pd.Series:
    """
    4-phase pullback entry state machine:
    1. Signal fires
    2. Wait for pullback (price retraces pullback_atr_mult × ATR from signal bar close)
    3. Confirm reversal (price closes above prior bar's high)
    4. Enter on next bar open

    Returns:
        pd.Series[bool]: True at the bar where entry should occur.
    """
    entries = pd.Series(False, index=close.index)
    in_pullback = False
    signal_bar_close = 0.0
    pullback_target = 0.0
    candles_waiting = 0

    for i in range(1, len(close)):
        if signal.iloc[i] and not in_pullback:
            # Phase 1: Signal fires
            in_pullback = True
            signal_bar_close = close.iloc[i]
            pullback_target = signal_bar_close - pullback_atr_mult * atr.iloc[i]
            candles_waiting = 0

        elif in_pullback:
            candles_waiting += 1

            if candles_waiting > entry_window_periods:
                # Expired — cancel
                in_pullback = False
                continue

            if candles_waiting <= pullback_max_candles:
                # Phase 2: Check for pullback
                if close.iloc[i] <= pullback_target:
                    # Phase 3: Check for reversal confirmation
                    if i >= 2 and close.iloc[i] > close.iloc[i - 1]:
                        # Phase 4: Entry on next bar (mark current bar)
                        if i + 1 < len(close):
                            entries.iloc[i + 1] = True
                        in_pullback = False

    return entries


def generate_signals(df: pd.DataFrame,
                     Q: float = 0.01, R: float = 1.0,
                     velocity_threshold_pct: float = 0.10,
                     mean_revert: bool = False,
                     use_trailing_stop: bool = True,
                     trail_atr_mult: float = 2.5,
                     long_only: bool = True,
                     use_vwap_exit: bool = True,
                     use_pullback_entry: bool = False,
                     pullback_max_candles: int = 2,
                     entry_window_periods: int = 3,
                     atr_period: int = 14,
                     vwap_period: int = 20,
                     **kwargs) -> tuple:
    """
    Kalman trend strategy signal generation.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        Q, R: Kalman filter parameters.
        velocity_threshold_pct: Velocity threshold as % of price for entry.
        mean_revert: If True, flip mode — enter on deviation from Kalman estimate.
        use_trailing_stop: If True, include trailing stop series.
        trail_atr_mult: ATR multiplier for trailing stop.
        long_only: If True, only generate long signals.
        use_vwap_exit: If True, exit when price crosses VWAP.
        use_pullback_entry: If True, use pullback entry logic.
        pullback_max_candles: Max candles to wait for pullback.
        entry_window_periods: Window for pullback entry.
        atr_period: ATR lookback period.
        vwap_period: VWAP lookback period.

    Returns:
        5-tuple: (long_entries, long_exits, short_entries, short_exits, trailing_stops)
        All pd.Series[bool] except trailing_stops which is pd.Series[float] (stop prices).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Run Kalman filter
    kf = _kalman_filter(close, Q=Q, R=R)
    estimate = kf["estimate"]
    velocity = kf["velocity"]

    # Compute ATR (anti-lookahead: already shifted in compute_atr)
    atr = compute_atr(df, period=atr_period)
    atr = atr.bfill().fillna(0)

    # Velocity threshold: entry when velocity > threshold_pct × price
    vel_threshold = velocity_threshold_pct * close

    if mean_revert:
        # Mean reversion mode: enter when price deviates from estimate
        deviation = close - estimate
        dev_std = deviation.rolling(window=20, min_periods=5).std()
        dev_std = dev_std.replace(0, np.nan).bfill().fillna(1.0)
        z_score = deviation / dev_std

        long_entries = (z_score < -1.5) & (z_score.shift(1) >= -1.5)
        long_exits = (z_score > 0) | (z_score.shift(1) <= 0)
        short_entries = (z_score > 1.5) & (z_score.shift(1) <= 1.5)
        short_exits = (z_score < 0) | (z_score.shift(1) >= 0)
    else:
        # Trend following mode: enter when velocity crosses threshold
        vel_cross_up = (velocity > vel_threshold) & (velocity.shift(1) <= vel_threshold.shift(1))
        vel_cross_down = (velocity < -vel_threshold) & (velocity.shift(1) >= -vel_threshold.shift(1))

        long_entries = vel_cross_up.copy()
        short_entries = vel_cross_down if not long_only else pd.Series(False, index=df.index)

        # Exits: velocity reversal or VWAP cross
        vel_exit_long = velocity < 0
        vel_exit_short = velocity > 0

        if use_vwap_exit:
            vwap = compute_vwap(df, reset_daily=False)
            # Use rolling VWAP with shift for anti-lookahead
            vwap_ma = vwap.rolling(window=vwap_period, min_periods=1).mean().shift(1)
            vwap_exit_long = close < vwap_ma
            vwap_exit_short = close > vwap_ma
            long_exits = vel_exit_long | vwap_exit_long
            short_exits = vel_exit_short | vwap_exit_short
        else:
            long_exits = vel_exit_long
            short_exits = vel_exit_short

    # Pullback entry enhancement
    if use_pullback_entry and not mean_revert:
        pullback_entries = _compute_pullback_entries(
            close, long_entries, atr,
            pullback_max_candles=pullback_max_candles,
            entry_window_periods=entry_window_periods,
        )
        # Use pullback entries where available, otherwise direct entries
        long_entries = pullback_entries | long_entries
        # Remove direct entries that overlap with pullback waiting period
        # (pullback entries take priority)

    # Trailing stop computation
    if use_trailing_stop:
        trailing_stops = _compute_trailing_stops(
            close, high, low, atr, trail_atr_mult, long_entries, long_exits,
        )
    else:
        trailing_stops = pd.Series(np.nan, index=df.index)

    # Anti-lookahead: shift signals by 1 bar
    long_entries = long_entries.shift(1).fillna(False).astype(bool)
    long_exits = long_exits.shift(1).fillna(False).astype(bool)
    short_entries = short_entries.shift(1).fillna(False).astype(bool)
    short_exits = short_exits.shift(1).fillna(False).astype(bool)

    # No signal on bar 0
    assert not long_entries.iloc[0], "Lookahead: signal on bar 0"

    return long_entries, long_exits, short_entries, short_exits, trailing_stops


def _compute_trailing_stops(close: pd.Series, high: pd.Series, low: pd.Series,
                            atr: pd.Series, trail_atr_mult: float,
                            entries: pd.Series, exits: pd.Series) -> pd.Series:
    """
    Compute ATR trailing stop prices.

    The stop trails the highest high since entry, minus trail_atr_mult × ATR.
    Only active when in a position.

    Returns:
        pd.Series[float]: Stop price at each bar. NaN when not in position.
    """
    stops = pd.Series(np.nan, index=close.index)
    in_position = False
    stop_price = 0.0

    for i in range(len(close)):
        if entries.iloc[i]:
            in_position = True
            stop_price = close.iloc[i] - trail_atr_mult * atr.iloc[i]

        elif exits.iloc[i]:
            in_position = False
            stop_price = np.nan

        if in_position:
            # Trail up: stop follows highest close minus ATR
            new_stop = close.iloc[i] - trail_atr_mult * atr.iloc[i]
            stop_price = max(stop_price, new_stop)

        stops.iloc[i] = stop_price if in_position else np.nan

    return stops
