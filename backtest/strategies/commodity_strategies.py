#!/usr/bin/env python3
"""All 21 trading strategies for commodities."""

import pandas as pd
import numpy as np


def _make_signals(df, entries_long, entries_short, exits_long, exits_short):
    """Convert boolean entry/exit masks into a signal DataFrame."""
    signals = pd.DataFrame({'timestamp': df['timestamp']})
    signal = np.zeros(len(df), dtype=int)
    
    position = 0
    for i in range(len(df)):
        if entries_long.iloc[i]:
            signal[i] = 1
            position = 1
        elif entries_short.iloc[i]:
            signal[i] = -1
            position = -1
        elif position == 1 and exits_long.iloc[i]:
            signal[i] = 0
            position = 0
        elif position == -1 and exits_short.iloc[i]:
            signal[i] = 0
            position = 0
        else:
            signal[i] = position
    
    signals['signal'] = signal
    return signals


def _trend_filter(df, period=200):
    """Return boolean: price above SMA = uptrend."""
    sma = df['close'].rolling(period).mean()
    return df['close'] > sma


# ============================================================
# TIER 1: TREND FOLLOWING
# ============================================================

def sma_crossover(df, fast=20, slow=50, trend_period=200):
    """SMA Crossover with trend filter."""
    df = df.copy()
    df['sma_fast'] = df['close'].rolling(fast).mean()
    df['sma_slow'] = df['close'].rolling(slow).mean()
    df['trend'] = _trend_filter(df, trend_period)
    
    cross_up = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1)) & df['trend']
    cross_down = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1)) & ~df['trend']
    exit_long = (df['sma_fast'] < df['sma_slow'])
    exit_short = (df['sma_fast'] > df['sma_slow'])
    
    return _make_signals(df, cross_up, cross_down, exit_long, exit_short)


def ema_crossover(df, fast=12, slow=26, trend_period=200):
    """EMA Crossover with trend filter."""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['trend'] = _trend_filter(df, trend_period)
    
    cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) & df['trend']
    cross_down = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) & ~df['trend']
    exit_long = df['ema_fast'] < df['ema_slow']
    exit_short = df['ema_fast'] > df['ema_slow']
    
    return _make_signals(df, cross_up, cross_down, exit_long, exit_short)


def donchian_breakout(df, period=20):
    """Donchian Channel Breakout."""
    df = df.copy()
    df['upper'] = df['high'].rolling(period).max().shift(1)
    df['lower'] = df['low'].rolling(period).min().shift(1)
    
    entries_long = df['close'] > df['upper']
    entries_short = df['close'] < df['lower']
    exit_long = df['close'] < df['lower']
    exit_short = df['close'] > df['upper']
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def turtle_trading(df, entry_period=20, exit_period=10):
    """Turtle Trading: 20-day breakout entry, 10-day breakout exit."""
    df = df.copy()
    df['entry_high'] = df['high'].rolling(entry_period).max().shift(1)
    df['entry_low'] = df['low'].rolling(entry_period).min().shift(1)
    df['exit_high'] = df['high'].rolling(exit_period).max().shift(1)
    df['exit_low'] = df['low'].rolling(exit_period).min().shift(1)
    
    entries_long = df['close'] > df['entry_high']
    entries_short = df['close'] < df['entry_low']
    exit_long = df['close'] < df['exit_low']
    exit_short = df['close'] > df['exit_high']
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def momentum_roc(df, roc_period=20, trend_period=200, threshold=0):
    """Momentum (Rate of Change) with trend filter."""
    df = df.copy()
    df['roc'] = df['close'].pct_change(roc_period) * 100
    df['trend'] = _trend_filter(df, trend_period)
    
    entries_long = (df['roc'] > threshold) & (df['roc'].shift(1) <= threshold) & df['trend']
    entries_short = (df['roc'] < -threshold) & (df['roc'].shift(1) >= -threshold) & ~df['trend']
    exit_long = df['roc'] < 0
    exit_short = df['roc'] > 0
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


# ============================================================
# TIER 2: MEAN REVERSION
# ============================================================

def bollinger_reversion(df, period=20, std_dev=2.0):
    """Bollinger Band Mean Reversion."""
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_mid'] + std_dev * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - std_dev * df['bb_std']
    
    entries_long = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
    entries_short = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
    exit_long = df['close'] > df['bb_mid']
    exit_short = df['close'] < df['bb_mid']
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def rsi_oversold_overbought(df, period=14, oversold=30, overbought=70):
    """RSI Oversold/Overbought."""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    entries_long = (df['rsi'] < oversold) & (df['rsi'].shift(1) >= oversold)
    entries_short = (df['rsi'] > overbought) & (df['rsi'].shift(1) <= overbought)
    exit_long = df['rsi'] > 50
    exit_short = df['rsi'] < 50
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def zscore_reversion(df, period=20, z_entry=2.0, z_exit=0.0):
    """Z-Score Mean Reversion."""
    df = df.copy()
    df['mean'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['zscore'] = (df['close'] - df['mean']) / df['std'].replace(0, np.nan)
    
    entries_long = (df['zscore'] < -z_entry) & (df['zscore'].shift(1) >= -z_entry)
    entries_short = (df['zscore'] > z_entry) & (df['zscore'].shift(1) <= z_entry)
    exit_long = df['zscore'] > z_exit
    exit_short = df['zscore'] < z_exit
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def vwap_reversion(df):
    """VWAP Reversion (session-based VWAP approximation)."""
    df = df.copy()
    # Approximate VWAP using rolling volume-weighted average
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        df['volume'] = 1  # fallback
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_std'] = typical_price.rolling(20).std()
    
    entries_long = (df['close'] < df['vwap'] - df['vwap_std']) & (df['close'].shift(1) >= (df['vwap'] - df['vwap_std']).shift(1))
    entries_short = (df['close'] > df['vwap'] + df['vwap_std']) & (df['close'].shift(1) <= (df['vwap'] + df['vwap_std']).shift(1))
    exit_long = df['close'] > df['vwap']
    exit_short = df['close'] < df['vwap']
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


# ============================================================
# TIER 3: BREAKOUT STRATEGIES
# ============================================================

def nr7_breakout(df):
    """NR7 Breakout: breakout after narrowest range in 7 bars."""
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['nr7'] = True
    for i in range(6, len(df)):
        window = df['range'].iloc[i-6:i+1]
        df.iloc[i, df.columns.get_loc('nr7')] = df['range'].iloc[i] == window.min()
    
    # Breakout on next bar after NR7
    df['nr7_shift'] = df['nr7'].shift(1)
    entries_long = df['nr7_shift'] & (df['close'] > df['high'].shift(1))
    entries_short = df['nr7_shift'] & (df['close'] < df['low'].shift(1))
    exit_long = df['close'] < df['low'].shift(2)
    exit_short = df['close'] > df['high'].shift(2)
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def volatility_breakout(df, lookback=20, contraction_mult=0.5, breakout_mult=1.5):
    """Volatility Breakout: entry after volatility contraction then expansion."""
    df = df.copy()
    df['atr'] = ((df['high'] - df['low']).rolling(lookback).mean())
    df['atr_ratio'] = (df['high'] - df['low']) / df['atr'].replace(0, np.nan)
    
    # Contraction followed by expansion
    contracted = df['atr_ratio'].shift(1) < contraction_mult
    expanded = df['atr_ratio'] > breakout_mult
    
    entries_long = contracted & expanded & (df['close'] > df['open'])
    entries_short = contracted & expanded & (df['close'] < df['open'])
    exit_long = df['atr_ratio'] < 1.0
    exit_short = df['atr_ratio'] < 1.0
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def opening_range_breakout(df, or_bars=2):
    """Opening Range Breakout (first N bars define range)."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    
    # Reset signals
    signals = pd.DataFrame({'timestamp': df['timestamp'], 'signal': 0})
    
    # For hourly data, use first 2 hours as OR
    position = 0
    or_high = 0
    or_low = float('inf')
    or_complete = False
    bars_in_session = 0
    
    for i in range(len(df)):
        hour = df.iloc[i]['hour']
        
        # Reset at start of day (00:00 or first bar)
        if hour == 0 or (i > 0 and df.iloc[i]['timestamp'].date() != df.iloc[i-1]['timestamp'].date()):
            bars_in_session = 0
            or_high = 0
            or_low = float('inf')
            or_complete = False
            position = 0
        
        bars_in_session += 1
        
        if bars_in_session <= or_bars:
            or_high = max(or_high, df.iloc[i]['high'])
            or_low = min(or_low, df.iloc[i]['low'])
            if bars_in_session == or_bars:
                or_complete = True
        elif or_complete:
            price = df.iloc[i]['close']
            if position == 0:
                if price > or_high:
                    signals.iloc[i, 1] = 1
                    position = 1
                elif price < or_low:
                    signals.iloc[i, 1] = -1
                    position = -1
            elif position == 1 and price < or_low:
                signals.iloc[i, 1] = 0
                position = 0
            elif position == -1 and price > or_high:
                signals.iloc[i, 1] = 0
                position = 0
            else:
                signals.iloc[i, 1] = position
    
    return signals


def session_breakout(df, session_hours=None):
    """Session Breakout: breakout of Asian/London/NY session ranges."""
    if session_hours is None:
        session_hours = {'asian': (0, 8), 'london': (8, 16), 'ny': (13, 21)}
    
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    
    signals = pd.DataFrame({'timestamp': df['timestamp'], 'signal': 0})
    position = 0
    session_high = 0
    session_low = float('inf')
    current_session = None
    
    for i in range(len(df)):
        hour = df.iloc[i]['hour']
        
        # Determine current session
        new_session = None
        for name, (start, end) in session_hours.items():
            if start <= hour < end:
                new_session = name
                break
        
        # Reset on new session
        if new_session != current_session:
            current_session = new_session
            session_high = df.iloc[i]['high']
            session_low = df.iloc[i]['low']
            position = 0
        else:
            session_high = max(session_high, df.iloc[i]['high'])
            session_low = min(session_low, df.iloc[i]['low'])
        
        price = df.iloc[i]['close']
        if position == 0:
            if price > session_high and current_session:
                signals.iloc[i, 1] = 1
                position = 1
            elif price < session_low and current_session:
                signals.iloc[i, 1] = -1
                position = -1
        elif position == 1 and price < session_low:
            signals.iloc[i, 1] = 0
            position = 0
        elif position == -1 and price > session_high:
            signals.iloc[i, 1] = 0
            position = 0
        else:
            signals.iloc[i, 1] = position
    
    return signals


# ============================================================
# TIER 4: PATTERN-BASED
# ============================================================

def inside_bar_breakout(df):
    """Inside Bar Breakout."""
    df = df.copy()
    is_inside = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    
    entries_long = is_inside.shift(1) & (df['close'] > df['high'].shift(1))
    entries_short = is_inside.shift(1) & (df['close'] < df['low'].shift(1))
    exit_long = df['close'] < df['low'].shift(2)
    exit_short = df['close'] > df['high'].shift(2)
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def engulfing_pattern(df):
    """Bullish/Bearish Engulfing Pattern."""
    df = df.copy()
    body = df['close'] - df['open']
    prev_body = body.shift(1)
    
    bullish_engulfing = (prev_body < 0) & (body > 0) & (df['open'] <= df['close'].shift(1)) & (df['close'] >= df['open'].shift(1))
    bearish_engulfing = (prev_body > 0) & (body < 0) & (df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1))
    
    entries_long = bullish_engulfing
    entries_short = bearish_engulfing
    exit_long = bearish_engulfing | (df['close'] < df['close'].shift(3))
    exit_short = bullish_engulfing | (df['close'] > df['close'].shift(3))
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


def pin_bar_reversal(df, wick_ratio=2.0):
    """Pin Bar Reversal: long lower wick = bullish, long upper wick = bearish."""
    df = df.copy()
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    bullish_pin = (lower_wick > wick_ratio * body) & (lower_wick > 0.6 * total_range) & (total_range > 0)
    bearish_pin = (upper_wick > wick_ratio * body) & (upper_wick > 0.6 * total_range) & (total_range > 0)
    
    entries_long = bullish_pin
    entries_short = bearish_pin
    exit_long = bearish_pin | (df['close'] < df['close'].shift(5))
    exit_short = bullish_pin | (df['close'] > df['close'].shift(5))
    
    return _make_signals(df, entries_long, entries_short, exit_long, exit_short)


# ============================================================
# TIER 5: MULTI-TIMEFRAME
# ============================================================

def multi_tf_momentum(df_daily, df_4h, df_1h):
    """Multi-Timeframe Momentum: Daily trend + 4H entry + 1H exit."""
    # Daily trend
    daily_trend = _trend_filter(df_daily, 50)
    daily_signals = df_daily[['timestamp']].copy()
    daily_signals['daily_trend'] = daily_trend.values
    
    # 4H momentum
    df_4h = df_4h.copy()
    df_4h['ema_fast'] = df_4h['close'].ewm(span=12).mean()
    df_4h['ema_slow'] = df_4h['close'].ewm(span=26).mean()
    df_4h['macd'] = df_4h['ema_fast'] - df_4h['ema_slow']
    df_4h['macd_signal'] = df_4h['macd'].ewm(span=9).mean()
    
    # Merge daily trend into 4H
    df_4h['date'] = df_4h['timestamp'].dt.date
    daily_signals['date'] = daily_signals['timestamp'].dt.date
    df_4h = df_4h.merge(daily_signals[['date', 'daily_trend']], on='date', how='left')
    df_4h['daily_trend'] = df_4h['daily_trend'].ffill().fillna(False)
    
    cross_up = (df_4h['macd'] > df_4h['macd_signal']) & (df_4h['macd'].shift(1) <= df_4h['macd_signal'].shift(1)) & df_4h['daily_trend']
    cross_down = (df_4h['macd'] < df_4h['macd_signal']) & (df_4h['macd'].shift(1) >= df_4h['macd_signal'].shift(1)) & ~df_4h['daily_trend']
    
    # 1H exit: EMA cross in opposite direction
    df_1h = df_1h.copy()
    df_1h['ema12'] = df_1h['close'].ewm(span=12).mean()
    df_1h['ema26'] = df_1h['close'].ewm(span=26).mean()
    
    signals = _make_signals(df_4h, cross_up, cross_down, 
                           df_4h['macd'] < df_4h['macd_signal'],
                           df_4h['macd'] > df_4h['macd_signal'])
    return signals


def multi_tf_mean_reversion(df_daily, df_4h):
    """Multi-Timeframe Mean Reversion: Daily trend + 4H oversold entry."""
    daily_trend = _trend_filter(df_daily, 50)
    daily_signals = df_daily[['timestamp']].copy()
    daily_signals['daily_trend'] = daily_trend.values
    
    df_4h = df_4h.copy()
    delta = df_4h['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_4h['rsi'] = 100 - (100 / (1 + rs))
    
    df_4h['date'] = df_4h['timestamp'].dt.date
    daily_signals['date'] = daily_signals['timestamp'].dt.date
    df_4h = df_4h.merge(daily_signals[['date', 'daily_trend']], on='date', how='left')
    df_4h['daily_trend'] = df_4h['daily_trend'].ffill().fillna(False)
    
    entries_long = (df_4h['rsi'] < 30) & (df_4h['rsi'].shift(1) >= 30) & df_4h['daily_trend']
    entries_short = (df_4h['rsi'] > 70) & (df_4h['rsi'].shift(1) <= 70) & ~df_4h['daily_trend']
    exit_long = df_4h['rsi'] > 50
    exit_short = df_4h['rsi'] < 50
    
    return _make_signals(df_4h, entries_long, entries_short, exit_long, exit_short)


# ============================================================
# TIER 6: SESSION-BASED
# ============================================================

def london_session_breakout(df):
    """London Session Breakout: breakout of London open range (8:00-10:00 UTC)."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    
    signals = pd.DataFrame({'timestamp': df['timestamp'], 'signal': 0})
    position = 0
    london_high = 0
    london_low = float('inf')
    london_range_set = False
    
    for i in range(len(df)):
        hour = df.iloc[i]['hour']
        new_day = i > 0 and df.iloc[i]['timestamp'].date() != df.iloc[i-1]['timestamp'].date()
        
        if new_day or hour == 8:
            london_high = 0
            london_low = float('inf')
            london_range_set = False
            position = 0
        
        if 8 <= hour <= 10:
            london_high = max(london_high, df.iloc[i]['high'])
            london_low = min(london_low, df.iloc[i]['low'])
            if hour == 10:
                london_range_set = True
        
        if london_range_set and hour > 10:
            price = df.iloc[i]['close']
            if position == 0:
                if price > london_high:
                    signals.iloc[i, 1] = 1
                    position = 1
                elif price < london_low:
                    signals.iloc[i, 1] = -1
                    position = -1
            elif position == 1 and price < london_low:
                signals.iloc[i, 1] = 0
                position = 0
            elif position == -1 and price > london_high:
                signals.iloc[i, 1] = 0
                position = 0
            else:
                signals.iloc[i, 1] = position
    
    return signals


def ny_session_momentum(df):
    """NY Session Momentum: continuation momentum during NY hours (13:00-21:00 UTC)."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['ema_fast'] = df['close'].ewm(span=8).mean()
    df['ema_slow'] = df['close'].ewm(span=21).mean()
    df['momentum'] = df['close'].pct_change(5) * 100
    
    signals = pd.DataFrame({'timestamp': df['timestamp'], 'signal': 0})
    position = 0
    
    for i in range(len(df)):
        hour = df.iloc[i]['hour']
        new_day = i > 0 and df.iloc[i]['timestamp'].date() != df.iloc[i-1]['timestamp'].date()
        
        if new_day:
            position = 0
        
        if 13 <= hour < 21:
            ema_bull = df.iloc[i]['ema_fast'] > df.iloc[i]['ema_slow']
            ema_bear = df.iloc[i]['ema_fast'] < df.iloc[i]['ema_slow']
            mom = df.iloc[i]['momentum']
            
            if position == 0:
                if ema_bull and mom > 0.1:
                    signals.iloc[i, 1] = 1
                    position = 1
                elif ema_bear and mom < -0.1:
                    signals.iloc[i, 1] = -1
                    position = -1
            elif position == 1 and (ema_bear or mom < -0.2):
                signals.iloc[i, 1] = 0
                position = 0
            elif position == -1 and (ema_bull or mom > 0.2):
                signals.iloc[i, 1] = 0
                position = 0
            else:
                signals.iloc[i, 1] = position
        else:
            # Close at end of NY session
            if position != 0:
                signals.iloc[i, 1] = 0
                position = 0
    
    return signals


def asian_range_breakout(df):
    """Asian Range Breakout: breakout of Asian session range (00:00-08:00 UTC)."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    
    signals = pd.DataFrame({'timestamp': df['timestamp'], 'signal': 0})
    position = 0
    asian_high = 0
    asian_low = float('inf')
    asian_range_set = False
    
    for i in range(len(df)):
        hour = df.iloc[i]['hour']
        new_day = i > 0 and df.iloc[i]['timestamp'].date() != df.iloc[i-1]['timestamp'].date()
        
        if new_day or hour == 0:
            asian_high = 0
            asian_low = float('inf')
            asian_range_set = False
            position = 0
        
        if 0 <= hour < 8:
            asian_high = max(asian_high, df.iloc[i]['high'])
            asian_low = min(asian_low, df.iloc[i]['low'])
            if hour == 7:
                asian_range_set = True
        
        if asian_range_set and hour >= 8:
            price = df.iloc[i]['close']
            if position == 0:
                if price > asian_high:
                    signals.iloc[i, 1] = 1
                    position = 1
                elif price < asian_low:
                    signals.iloc[i, 1] = -1
                    position = -1
            elif position == 1 and price < asian_low:
                signals.iloc[i, 1] = 0
                position = 0
            elif position == -1 and price > asian_high:
                signals.iloc[i, 1] = 0
                position = 0
            else:
                signals.iloc[i, 1] = position
    
    return signals


# ============================================================
# STRATEGY REGISTRY
# ============================================================

SINGLE_TF_STRATEGIES = {
    "SMA Crossover": sma_crossover,
    "EMA Crossover": ema_crossover,
    "Donchian Breakout": donchian_breakout,
    "Turtle Trading": turtle_trading,
    "Momentum ROC": momentum_roc,
    "Bollinger Reversion": bollinger_reversion,
    "RSI Oversold/Overbought": rsi_oversold_overbought,
    "Z-Score Reversion": zscore_reversion,
    "VWAP Reversion": vwap_reversion,
    "NR7 Breakout": nr7_breakout,
    "Volatility Breakout": volatility_breakout,
    "Opening Range Breakout": opening_range_breakout,
    "Session Breakout": session_breakout,
    "Inside Bar Breakout": inside_bar_breakout,
    "Engulfing Pattern": engulfing_pattern,
    "Pin Bar Reversal": pin_bar_reversal,
    "London Session Breakout": london_session_breakout,
    "NY Session Momentum": ny_session_momentum,
    "Asian Range Breakout": asian_range_breakout,
}

MULTI_TF_STRATEGIES = {
    "Multi-TF Momentum": multi_tf_momentum,
    "Multi-TF Mean Reversion": multi_tf_mean_reversion,
}
