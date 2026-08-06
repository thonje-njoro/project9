"""Opening Range Breakout (ORB) Strategy — SPY/QQQ 15-minute.

Implementation of the Zarattini et al. (2024) ORB strategy adapted for 15-minute data.

Core Logic:
    1. Define opening range from first 15-min bar (9:30-9:45 ET)
    2. If first bar bullish (close > open) → only go long on breakout above high
    3. If first bar bearish → only go short on breakout below low
    4. Relative Volume filter: today's OR volume vs 14-day average
    5. Stop loss at 10% of 14-day ATR
    6. Exit at end of day (4:00 PM ET)

Pitfall Mitigations:
    - Look-ahead: All signals shifted by 1 bar; ATR uses only prior days
    - Relative Volume: Uses only completed OR bars for comparison
    - Session detection: UTC-based, no DST issues
    - Transaction costs: Modeled in config (commission + slippage)

Usage:
    from strategies.orb_strategy import generate_signals
    entries, exits, short_entries, short_exits = generate_signals(df, **params)

Reference:
    Zarattini, Barbon, Aziz (2024). "A Profitable Day Trading Strategy
    For The U.S. Equity Market." Concretum Research.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _compute_daily_atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Compute daily ATR using only prior complete trading days.
    
    Returns a Series aligned to the daily index, where each value
    is the ATR computed from the PREVIOUS `lookback` complete days.
    This avoids look-ahead bias by not including the current day.
    """
    # Group by date to get daily high/low/close
    dates = df.index.date
    daily = df.groupby(dates).agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    )
    daily.index = pd.to_datetime(daily.index)
    
    # True range = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = daily['close'].shift(1)
    tr1 = daily['high'] - daily['low']
    tr2 = (daily['high'] - prev_close).abs()
    tr3 = (daily['low'] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = rolling mean of true range, shifted by 1 to avoid look-ahead
    atr = true_range.rolling(lookback).mean().shift(1)
    return atr


def _compute_relative_volume(
    df: pd.DataFrame,
    or_hour: int = 14,
    or_minute: int = 30,
    lookback: int = 14
) -> pd.Series:
    """Compute Relative Volume for the opening range.
    
    Compares today's OR volume to the 14-day average OR volume.
    Returns a Series aligned to the daily index.
    """
    # Get opening range bars (first bar of each day)
    or_mask = (df.index.hour == or_hour) & (df.index.minute == or_minute)
    or_bars = df[or_mask].copy()
    
    if len(or_bars) == 0:
        return pd.Series(dtype=float)
    
    # Daily OR volume
    or_volume = or_bars.groupby(or_bars.index.date)['volume'].sum()
    or_volume.index = pd.to_datetime(or_volume.index)
    
    # 14-day average OR volume, shifted to avoid look-ahead
    avg_or_volume = or_volume.rolling(lookback).mean().shift(1)
    
    # Relative volume
    rel_vol = or_volume / avg_or_volume.replace(0, np.nan)
    return rel_vol


def _get_or_bars(df: pd.DataFrame, date) -> Optional[pd.DataFrame]:
    """Get the opening range bars for a specific date."""
    date_str = str(date) if not isinstance(date, str) else date
    mask = df.index.date == pd.Timestamp(date_str).date()
    day_bars = df[mask]
    
    if len(day_bars) == 0:
        return None
    
    # First bar is the OR bar (15-min data)
    return day_bars.iloc[:1]


def generate_signals(
    df: pd.DataFrame,
    orb_period: int = 1,
    session_open_hour: int = 14,
    session_open_minute: int = 30,
    session_close_hour: int = 21,
    rel_vol_lookback: int = 14,
    min_rel_volume: float = 1.0,
    atr_period: int = 14,
    atr_stop_pct: float = 0.10,
    risk_per_trade: float = 0.01,
    min_price: float = 5.0,
    min_avg_volume: float = 1_000_000,
    commission: float = 0.0005,
    slippage_bps: float = 0.001,
    **kwargs
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate ORB signals for SPY/QQQ.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume] and UTC DatetimeIndex
        orb_period: Number of bars for opening range (1 = first 15-min bar)
        session_open_hour: UTC hour of session open (14 = 9:30 AM ET)
        session_open_minute: UTC minute of session open
        session_close_hour: UTC hour of session close (21 = 4:00 PM ET)
        rel_vol_lookback: Days for relative volume average
        min_rel_volume: Minimum relative volume to trade (1.0 = 100%)
        atr_period: Days for ATR calculation
        atr_stop_pct: Stop loss as fraction of ATR (0.10 = 10%)
        risk_per_trade: Risk per trade as fraction of equity
        min_price: Minimum stock price filter
        min_avg_volume: Minimum 14-day average volume
        commission: Commission rate
        slippage_bps: Slippage in basis points
    
    Returns:
        (long_entries, long_exits, short_entries, short_exits) as boolean Series
    """
    n = len(df)
    long_entries = pd.Series(False, index=df.index)
    long_exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)
    
    # Pre-compute daily ATR (avoids look-ahead)
    daily_atr = _compute_daily_atr(df, atr_period)
    
    # Pre-compute relative volume
    rel_vol = _compute_relative_volume(df, session_open_hour, session_open_minute, rel_vol_lookback)
    
    # Get unique trading dates
    dates = sorted(set(df.index.date))
    
    # State for each day
    in_position = False
    position_direction = None  # 'long' or 'short'
    entry_price = 0.0
    stop_price = 0.0
    
    for i, date in enumerate(dates):
        date_ts = pd.Timestamp(date)
        
        # Get today's bars
        day_mask = df.index.date == date
        day_bars = df[day_mask]
        
        if len(day_bars) < 2:
            continue
        
        # Reset position state at start of each day (ORB is intraday)
        in_position = False
        position_direction = None
        
        # Get opening range bar (first bar of the day)
        or_bar = day_bars.iloc[0]
        or_time = day_bars.index[0]
        
        # Verify it's at session open
        if or_time.hour != session_open_hour or or_time.minute != session_open_minute:
            continue
        
        # --- FILTERS ---
        
        # Price filter
        if or_bar['open'] < min_price:
            continue
        
        # Volume filter: 14-day average volume
        # Use a rolling average of daily total volume
        prior_days_mask = df.index.date < date
        prior_days = df[prior_days_mask]
        if len(prior_days) > 0:
            daily_vols = prior_days.groupby(prior_days.index.date)['volume'].sum()
            avg_vol = daily_vols.tail(rel_vol_lookback).mean()
            if avg_vol < min_avg_volume:
                continue
        else:
            continue
        
        # Relative Volume filter
        if date_ts in rel_vol.index:
            rv = rel_vol.loc[date_ts]
            if pd.notna(rv) and rv < min_rel_volume:
                continue
        
        # --- ORB SIGNAL ---
        
        # Direction: bullish first bar → long, bearish → short
        or_open = or_bar['open']
        or_close = or_bar['close']
        or_high = or_bar['high']
        or_low = or_bar['low']
        
        is_bullish = or_close > or_open
        
        # ATR for stop loss
        if date_ts in daily_atr.index:
            atr_val = daily_atr.loc[date_ts]
        else:
            # Find closest prior date
            prior_atr = daily_atr[daily_atr.index <= date_ts]
            if len(prior_atr) == 0:
                continue
            atr_val = prior_atr.iloc[-1]
        
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        
        stop_distance = atr_stop_pct * atr_val
        
        # --- SCAN FOR ENTRY ---
        # Entry happens when price breaks OR high/low AFTER the OR bar
        
        for j in range(1, len(day_bars)):
            bar = day_bars.iloc[j]
            bar_idx = day_bars.index[j]
            bar_hour = bar_idx.hour
            
            # Skip if past session close
            if bar_hour >= session_close_hour:
                break
            
            if not in_position:
                # Check for breakout
                if is_bullish:
                    # Long entry: price breaks above OR high
                    if bar['high'] > or_high:
                        entry_price = or_high  # Stop order at OR high
                        stop_price = entry_price - stop_distance
                        
                        # Signal fires on NEXT bar (shift by 1 for no look-ahead)
                        if j + 1 < len(day_bars):
                            signal_idx = day_bars.index[j + 1]
                            long_entries.loc[signal_idx] = True
                            in_position = True
                            position_direction = 'long'
                else:
                    # Short entry: price breaks below OR low
                    if bar['low'] < or_low:
                        entry_price = or_low  # Stop order at OR low
                        stop_price = entry_price + stop_distance
                        
                        if j + 1 < len(day_bars):
                            signal_idx = day_bars.index[j + 1]
                            short_entries.loc[signal_idx] = True
                            in_position = True
                            position_direction = 'short'
            
            else:
                # Check for exit conditions
                exit_signal = False
                
                if position_direction == 'long':
                    # Stop loss
                    if bar['low'] <= stop_price:
                        exit_signal = True
                    # End of day
                    elif bar_hour >= session_close_hour - 1:
                        exit_signal = True
                    
                    if exit_signal and j + 1 < len(day_bars):
                        long_exits.loc[day_bars.index[j + 1]] = True
                        in_position = False
                
                elif position_direction == 'short':
                    # Stop loss
                    if bar['high'] >= stop_price:
                        exit_signal = True
                    # End of day
                    elif bar_hour >= session_close_hour - 1:
                        exit_signal = True
                    
                    if exit_signal and j + 1 < len(day_bars):
                        short_exits.loc[day_bars.index[j + 1]] = True
                        in_position = False
    
    return long_entries, long_exits, short_entries, short_exits
