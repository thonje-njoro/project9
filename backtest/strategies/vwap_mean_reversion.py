"""
VWAP Mean Reversion Strategy — Intraday VWAP Z-score framework.

Mathematical Formulation:
─────────────────────────

Session-Intraday VWAP (daily reset):
    VWAP_t = Σ(P_i × V_i) / Σ(V_i)    for i ∈ [session_open, t]

    where P_i = (H_i + L_i + C_i) / 3  (typical price)

Rolling VWAP (configurable window):
    VWAP_t = Σ(P_i × V_i) / Σ(V_i)    for i ∈ [t-window+1, t]

Distance Metric (Z-score):
    Z_t = (P_t - VWAP_t) / σ_t

    where σ_t = rolling std(P_i - VWAP_i) over lookback period

Signals:
    Long entry:  Z_t < -z_entry     (price deeply below VWAP → buy)
    Short entry: Z_t > +z_entry     (price far above VWAP → sell)
    Long exit:   Z_t ≥ -z_exit      (reversion to VWAP midline)
    Short exit:  Z_t ≤ +z_exit

Additional exit rules:
    - Time stop: exit after max_hold_bars if no reversion occurred
    - Hard ATR stop: exit if price moves > trail_atr_mult × ATR against position
    - Session end: flatten before market close (configurable)

References:
    - "Volume-Weighted Average Price (VWAP)" — Bialkowski et al. (2012)
    - "Intraday Momentum and Reversal" — Gao et al. (2018)
    - "Optimal Mean Reversion Trading" — arXiv:1411.5062
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Core VWAP Calculations
# ──────────────────────────────────────────────────────────────────────────────


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute daily session-resetting intraday VWAP.

    Resets at each calendar day boundary. Uses typical price:
        P_t = (H_t + L_t + C_t) / 3

    Returns:
        VWAP series, same index as df. NaNs for first bar of each session.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]

    # Identify trading days
    if hasattr(df.index, "date"):
        dates = df.index.date
    elif hasattr(df.index, "normalize"):
        dates = df.index.normalize()
    else:
        dates = df.index

    # Groupby date for session-resetting cumulative sums
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()

    vwap = cum_pv / cum_vol.replace(0, float("nan"))
    # First bar of each session → NaN (no prior vol), bfill is wrong here
    # Instead, fill with the typical price itself for that bar
    vwap = vwap.fillna(tp)
    return vwap


def compute_rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute rolling VWAP over a fixed bar window.

    VWAP_t = sum(P_i * V_i for i in [t-window+1, t]) / sum(V_i for i ...)

    This is the standard anchored rolling version, not session-resetting.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    vwap = pv.rolling(window).sum() / df["volume"].rolling(window).sum()
    return vwap.bfill().fillna(tp)


def compute_vwap_distance(
    df: pd.DataFrame,
    use_daily_vwap: bool = True,
    vwap_window: int = 20,
    z_score_lookback: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute VWAP anchor, distance (Z-score), and rolling std.

    Args:
        df: OHLCV DataFrame with 'close', 'high', 'low', 'volume'
        use_daily_vwap: True = session-resetting intraday VWAP
                         False = rolling window VWAP
        vwap_window: Rolling window for VWAP (ignored if use_daily_vwap=True)
        z_score_lookback: Lookback for rolling std of VWAP distance

    Returns:
        (vwap, z_score, vwap_std)
            vwap: VWAP anchor line
            z_score: Standardized distance from VWAP
            vwap_std: Rolling std of (close - vwap) distance
    """
    close = df["close"]

    if use_daily_vwap:
        vwap = compute_intraday_vwap(df)
    else:
        vwap = compute_rolling_vwap(df, window=vwap_window)

    # Distance in price units
    distance = close - vwap

    # Rolling standard deviation of the distance
    # This normalizes the Z-score so it's adaptive to intraday volatility
    vwap_std = distance.rolling(z_score_lookback, min_periods=5).std()
    vwap_std = vwap_std.replace(0, float("nan")).bfill().fillna(
        close.pct_change().rolling(20).std() * close
    )

    z_score = distance / vwap_std
    return vwap, z_score, vwap_std


# ──────────────────────────────────────────────────────────────────────────────
# Relative Volume Filter (from Zarattini, Barbon & Aziz 2024)
# ──────────────────────────────────────────────────────────────────────────────


def compute_relative_volume(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute Relative Volume: opening range volume vs its multi-day average.

    For each trading day, takes the volume of the first N-minute bar (the
    "opening range" volume) and divides it by the rolling average of first-bar
    volumes over `period` previous trading days.

    This measures whether a stock is "in play" — abnormally high opening volume
    signals a catalyst-driven trend day where mean reversion fails. Values near
    1.0 mean normal volume (good for MR), values > 1.5-2.0 suggest a trend day.

    From Zarattini, Barbon & Aziz (2024):
      "A portfolio of top 20 Stocks in Play (RelVol > 1.0) achieved Sharpe 2.81,
       annual alpha 36%. Conversely, low RelVol stocks are ideal for mean
       reversion strategies."

    Returns:
        Series of relative-volume ratios, one per bar (forward-filled intraday).
        On days with no data, value is NaN.
    """
    close = df["close"]
    idx = df.index

    # Identify trading days
    dates = idx.date if hasattr(idx, "date") else idx.normalize()

    # First bar of each day (the "opening range" bar)
    # We detect day boundaries where the date changes
    date_unique = pd.Series(dates, index=idx).unique()

    first_bar_vol = pd.Series(0.0, index=idx, dtype=float)
    for d in date_unique:
        day_mask = pd.Series(dates, index=idx) == d
        day_indices = idx[day_mask.values]
        if len(day_indices) > 0:
            first_bar_idx = day_indices[0]
            first_bar_vol.loc[first_bar_idx] = df.loc[first_bar_idx, "volume"]

    # Rolling average of first-bar volumes over `period` days
    # We only have one first-bar entry per day, so group and compute
    first_bar_by_day = first_bar_vol.groupby(dates).first()
    avg_first_bar_vol = first_bar_by_day.rolling(period, min_periods=5).mean()

    # Build relative volume series: current first-bar vol / avg first-bar vol
    rel_vol_by_day = first_bar_by_day / avg_first_bar_vol.replace(0, float("nan"))

    # Map back to every bar: fill each trading day with its single rel_vol value
    rel_vol = pd.Series(float("nan"), index=idx)
    for d in date_unique:
        day_mask = pd.Series(dates, index=idx) == d
        if d in rel_vol_by_day.index and pd.notna(rel_vol_by_day[d]):
            rel_vol.loc[day_mask.values] = rel_vol_by_day[d]

    return rel_vol.ffill()


def generate_signals(
    df: pd.DataFrame,
    # ── VWAP Configuration ──
    use_daily_vwap: bool = True,
    vwap_window: int = 20,
    z_score_lookback: int = 20,
    # ── Entry / Exit Thresholds ──
    z_entry: float = 2.5,
    z_exit: float = 0.0,
    # ── Mean-reversion confirmation ──
    require_reversal: bool = True,
    reversal_lookback: int = 3,
    # ── Risk Controls ──
    use_trailing_stop: bool = True,
    trail_atr_mult: float = 2.0,
    max_hold_bars: int = 48,
    use_time_stop: bool = True,
    long_only: bool = True,
    # ── Regime filter ──
    min_volume_ratio: float = 0.3,
    skip_trend_days: bool = False,
    trend_window: int = 20,
    adx_threshold: float = 25.0,
    # ── Relative Volume filter (Zarattini, Barbon & Aziz 2024) ──
    use_relative_volume: bool = True,
    max_relative_volume: float = 1.5,
    rel_volume_period: int = 14,
    # ── ATR Change Filter (from Sunrise Ogle / FORWIZ AI) ──
    use_atr_change_filter: bool = False,
    max_atr_change: float = 0.05,
    # ── Time-of-Day Filter (from Sunrise Ogle / FORWIZ AI) ──
    use_time_filter: bool = False,
    entry_start_hour_et: int = 9,
    entry_end_hour_et: int = 16,
    # ── Take Profit (from Sunrise Ogle OCA architecture) ──
    use_take_profit: bool = False,
    atr_tp_mult: float = 3.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """VWAP Mean Reversion signal generator.

    Entry:
      Long:  Z < -z_entry  (price far below VWAP — mean reversion pull)
      Short: Z > +z_entry  (price far above VWAP — mean reversion pull)

      Optional: require a reversal confirmation over reversal_lookback bars
      (price must have moved back toward VWAP in recent bars — avoids catching
       a knife on trend days).

    Exit:
      Primary: Z >= -z_exit (long) / Z <= +z_exit (short) — reversion to VWAP
      Time stop: Exit after max_hold_bars with no reversion
      ATR trailing: Hard stop at trail_atr_mult × ATR

    Returns:
        (long_entries, long_exits, short_entries, short_exits, trailing_stops)
    """
    close = df["close"]
    idx = df.index
    n = len(df)

    if n < 30:
        empty = pd.Series(False, index=idx)
        zeros = pd.Series(0.0, index=idx)
        return (empty, empty, empty, empty, zeros)

    # ── 1. Compute VWAP and Z-score ──
    vwap, z_score, vwap_std = compute_vwap_distance(
        df,
        use_daily_vwap=use_daily_vwap,
        vwap_window=vwap_window,
        z_score_lookback=z_score_lookback,
    )

    # ── 2. Volume filter ──
    vol_ma = df["volume"].rolling(vwap_window).mean()
    vol_ratio = df["volume"] / vol_ma.replace(0, float("nan"))
    vol_ok = vol_ratio.fillna(0) >= min_volume_ratio

    # ── 3. ADX regime filter (optional: skip strong trend days) ──
    if skip_trend_days:
        # Compute directional movement
        high, low = df["high"], df["low"]
        up_move = high.diff()
        down_move = low.diff() * -1

        plus_dm = pd.Series(0.0, index=idx)
        minus_dm = pd.Series(0.0, index=idx)

        up_stronger = (up_move > down_move) & (up_move > 0)
        down_stronger = (down_move > up_move) & (down_move > 0)
        plus_dm[up_stronger] = up_move[up_stronger]
        minus_dm[down_stronger] = down_move[down_stronger]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14).mean().replace(0, float("nan"))
        plus_di = 100 * plus_dm.rolling(14).sum() / atr_14
        minus_di = 100 * minus_dm.rolling(14).sum() / atr_14
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))) * 100
        adx = dx.rolling(trend_window).mean().fillna(0)

        trend_ok = adx < adx_threshold  # low ADX → ranging → good for MR
    else:
        trend_ok = pd.Series(True, index=idx)

    # ── 4. Relative Volume filter (Zarattini, Barbon & Aziz 2024) ──
    # High opening-range volume signals catalyst-driven trend days where
    # mean reversion fails. Skip trades when RelVol is abnormally high.
    if use_relative_volume:
        rel_vol = compute_relative_volume(df, period=rel_volume_period)
        # Cap to avoid blowup from extreme values; NaN → allow trade
        rel_vol = rel_vol.fillna(0.0).clip(0.0, 100.0)
        rel_vol_ok = rel_vol < max_relative_volume
        # Also: require rel_vol to be computable (>5 days of data)
        min_days_ok = rel_vol.notna().sum() > 5
        if not min_days_ok:
            rel_vol_ok = pd.Series(True, index=idx)
    else:
        rel_vol_ok = pd.Series(True, index=idx)

    # ── 5. ATR Change Filter (from Sunrise Ogle / FORWIZ AI) ──
    # Volatility should be stable or contracting for mean reversion.
    # Rapid ATR expansion signals a volatility event (earnings, macro gap)
    # where price is unlikely to revert within the expected timeframe.
    atr_change_ok = pd.Series(True, index=idx)
    if use_atr_change_filter:
        from risk.position_sizer import compute_atr
        atr = compute_atr(df, 14)
        atr_change = atr.diff().abs().fillna(0)
        atr_change_ok = atr_change < max_atr_change * close * 0.01  # scale by price

    # ── 6. Time-of-Day Filter (from Sunrise Ogle / FORWIZ AI) ──
    # Mean reversion works best during regular market hours when liquidity
    # is sufficient. Pre-market, after-hours, and lunch-hour thinness
    # produce false signals.
    time_ok = pd.Series(True, index=idx)
    if use_time_filter:
        try:
            import pytz
            et_tz = pytz.timezone("US/Eastern")
            # Index may be already tz-aware in UTC
            if df.index.tz is None:
                local_idx = df.index.tz_localize("UTC").tz_convert(et_tz)
            else:
                local_idx = df.index.tz_convert(et_tz)
            entry_hour = local_idx.hour
            entry_minute = local_idx.minute
            start_mins = entry_start_hour_et * 60
            end_mins = entry_end_hour_et * 60
            current_mins = entry_hour * 60 + entry_minute
            time_ok = (current_mins >= start_mins) & (current_mins <= end_mins)
        except ImportError:
            pass

    # ── 7. Raw entry signals ──
    raw_long_entry = (z_score < -z_entry) & vol_ok & trend_ok & rel_vol_ok & atr_change_ok & time_ok
    raw_short_entry = (z_score > z_entry) & vol_ok & trend_ok & rel_vol_ok & atr_change_ok & time_ok

    # ── 8. Reversal confirmation (optional) ──
    if require_reversal and reversal_lookback > 0:
        # Price must have moved back toward VWAP in recent bars
        # i.e., the deviation is narrowing
        dist = close - vwap
        dist_change = dist.diff(reversal_lookback)
        # For long entries: distance is becoming less negative (reverting up)
        # For short entries: distance is becoming less positive (reverting down)
        long_reversal = dist_change > 0  # moving up toward VWAP
        short_reversal = dist_change < 0  # moving down toward VWAP

        long_entries = raw_long_entry & long_reversal
        short_entries = raw_short_entry & short_reversal
    else:
        long_entries = raw_long_entry
        short_entries = raw_short_entry

    # ── 9. Exit signals ──
    # Primary: Z-score reversion to VWAP midline
    if z_exit == 0.0:
        long_exits = z_score >= 0.0  # crossed back to/above VWAP
        short_exits = z_score <= 0.0  # crossed back to/below VWAP
    else:
        long_exits = z_score >= -z_exit  # reverted to within exit threshold
        short_exits = z_score <= z_exit

    # ── 10. Take-Profit Exit (from Sunrise Ogle OCA architecture) ──
    # After entry, if price moves ATR * atr_tp_mult in the right direction,
    # take profit. For MR, this captures outlier moves that revert completely
    # through VWAP and beyond.
    if use_take_profit and atr_tp_mult > 0:
        from risk.position_sizer import compute_atr
        atr_local = compute_atr(df, 14)
        tp_long = close > vwap + atr_local * atr_tp_mult  # price rose above VWAP by enough
        tp_short = close < vwap - atr_local * atr_tp_mult
        long_exits = long_exits | tp_long
        short_exits = short_exits | tp_short

    # ── 7. Time stop ──
    if use_time_stop and max_hold_bars > 0:
        # Track bars since entry using forward-fill of entry signals
        in_long = long_entries.astype(int).where(long_entries, 0)
        in_short = short_entries.astype(int).where(short_entries, 0)

        # Cumsum of entries resets on exit: build a simple position counter
        long_pos = pd.Series(0, index=idx)
        short_pos = pd.Series(0, index=idx)

        # Vectorized: count bars since last entry
        long_hold = pd.Series(0, index=idx)
        short_hold = pd.Series(0, index=idx)

        hold_counter = 0
        in_long_pos = False
        in_short_pos = False

        for i in range(len(df)):
            if long_entries.iloc[i] and not in_long_pos:
                in_long_pos = True
                hold_counter = 0
            elif short_entries.iloc[i] and not in_short_pos:
                in_short_pos = True
                hold_counter = 0

            if in_long_pos:
                long_hold.iloc[i] = hold_counter
                if long_exits.iloc[i]:
                    in_long_pos = False
                    hold_counter = 0
                else:
                    hold_counter += 1
            elif in_short_pos:
                short_hold.iloc[i] = hold_counter
                if short_exits.iloc[i]:
                    in_short_pos = False
                    hold_counter = 0
                else:
                    hold_counter += 1
            else:
                hold_counter = 0

        # Force exit on time stop
        time_stop_long = (long_hold >= max_hold_bars) & in_long_pos
        time_stop_short = (short_hold >= max_hold_bars) & in_short_pos

        long_exits = long_exits | time_stop_long
        short_exits = short_exits | time_stop_short

    # ── 8. Trailing stop (ATR-based) ──
    trailing_stops = pd.Series(0.0, index=idx)
    if use_trailing_stop and trail_atr_mult > 0:
        from risk.position_sizer import compute_atr
        atr = compute_atr(df, 14)
        trailing_stops = atr * trail_atr_mult

    # ── 9. Long-only restriction ──
    if long_only:
        short_entries = pd.Series(False, index=idx)
        short_exits = pd.Series(False, index=idx)

    return (
        long_entries.shift(1).fillna(False).infer_objects(copy=False),
        long_exits.shift(1).fillna(False).infer_objects(copy=False),
        short_entries.shift(1).fillna(False).infer_objects(copy=False),
        short_exits.shift(1).fillna(False).infer_objects(copy=False),
        trailing_stops.shift(1).fillna(0).infer_objects(copy=False),
    )
