"""Entry condition filters to improve win rate.

Includes adaptive indicators from aligrithm.com research:
- Adaptive RSI: lookback tuned to measured dominant cycle
- Convolution detector: Pearson folding for reversal timing

Includes multi-source data filters from TradingAgents:
- Sentiment filter: blocks entries during negative sentiment
- Fundamentals filter: blocks entries with weak fundamentals
- Insider filter: blocks entries during heavy insider selling
- Reflection filter: blocks entries based on historical trade lessons
"""

import numpy as np
import pandas as pd
from typing import Optional


def sentiment_filter(df: pd.DataFrame, ticker: str = "SPY", min_score: float = -0.5) -> pd.Series:
    from optimization.sentiment_filter import sentiment_filter as _sf
    return _sf(df, ticker, min_score)


def fundamentals_filter(df: pd.DataFrame, ticker: str = "SPY", min_fcf_yield: float = 0.0, max_debt_ratio: float = 1.0) -> pd.Series:
    from optimization.sentiment_filter import fundamentals_filter as _ff
    return _ff(df, ticker, min_fcf_yield, max_debt_ratio)


def insider_filter(df: pd.DataFrame, ticker: str = "SPY", min_net_sentiment: float = -0.5) -> pd.Series:
    from optimization.sentiment_filter import insider_filter as _if
    return _if(df, ticker, min_net_sentiment)


def reflection_filter(df: pd.DataFrame, symbol: str = "SPY", regime: Optional[str] = None) -> pd.Series:
    from optimization.reflection_filter import reflection_filter as _rf
    return _rf(df, symbol, regime)


def trend_filter(df: pd.DataFrame, period: int = 200) -> tuple[pd.Series, pd.Series]:
    sma = df["close"].rolling(period).mean()
    allow_long = df["close"] > sma
    allow_short = df["close"] < sma
    return allow_long.shift(1).fillna(False), allow_short.shift(1).fillna(False)


def volatility_regime_filter(df: pd.DataFrame) -> pd.Series:
    log_returns = np.log(df["close"] / df["close"].shift(1))
    realized_vol = log_returns.rolling(20).std()
    vol_252_high = realized_vol.rolling(252).quantile(0.80)
    vol_252_low = realized_vol.rolling(252).quantile(0.10)
    allow = (realized_vol < vol_252_high) & (realized_vol > vol_252_low)
    return allow.shift(1).fillna(False)


def rsi_filter(
    df: pd.DataFrame, period: int = 14, oversold: float = 40, overbought: float = 60
) -> tuple[pd.Series, pd.Series]:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return (rsi < oversold).shift(1).fillna(False), (rsi > overbought).shift(1).fillna(False)


def adaptive_rsi_filter(
    df: pd.DataFrame,
    oversold: float = 40,
    overbought: float = 60,
    min_period: int = 8,
    max_period: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """
    RSI with lookback tuned to half the measured dominant cycle.

    From aligrithm.com: 'Adaptive Indicators: Tuning RSI to the Measured Cycle'.
    Uses subsampled cycle measurement for speed, then interpolates back.
    """
    close = df["close"]
    delta = close.diff()

    step = 4
    sub_close = close.iloc[::step].values
    sub_n = len(sub_close)
    test_lags = np.arange(min_period, max_period + 1, 8)
    sub_best_lag = np.full(sub_n, min_period, dtype=float)

    detrend_w = min(max_period, sub_n)
    for end in range(detrend_w, sub_n):
        window = sub_close[end - detrend_w:end].copy()
        window -= window.mean()
        norm = np.dot(window, window)
        if norm < 1e-12:
            continue
        best_corr = -1.0
        bl = min_period
        for lag in test_lags:
            if lag >= len(window) // 2:
                break
            corr = np.dot(window[lag:], window[:-lag]) / norm
            if corr > best_corr:
                best_corr = corr
                bl = lag
        sub_best_lag[end] = bl

    sub_idx = np.arange(0, len(close), step)[:sub_n]
    full_best_lag = np.interp(np.arange(len(close)), sub_idx, sub_best_lag)
    full_best_lag[:detrend_w] = min_period

    adaptive_period = np.clip(full_best_lag / 2, min_period, max_period).astype(int)
    rsi = pd.Series(50.0, index=df.index)

    for period_val in np.unique(adaptive_period):
        if period_val < min_period:
            continue
        mask = adaptive_period == period_val
        gain = delta.clip(lower=0).rolling(period_val).mean()
        loss = (-delta.clip(upper=0)).rolling(period_val).mean()
        rs = gain / loss.replace(0, np.nan)
        full_rsi = 100 - (100 / (1 + rs))
        rsi.values[mask] = full_rsi.values[mask]

    return (rsi < oversold).shift(1).fillna(False), (rsi > overbought).shift(1).fillna(False)




def convolution_reversal_filter(
    df: pd.DataFrame,
    fold_half: int = 10,
    min_correlation: float = 0.6,
) -> pd.Series:
    """
    Detect symmetric turning points via Pearson correlation folding.

    From aligrithm.com 'Convolution: Detecting Reversals by Folding Price':
    Fold price about candidate bar into forward/reversed halves.
    High negative correlation between halves indicates a local extremum
    (potential reversal point). This avoids moving-average lag.

    Vectorized implementation using rolling window correlation.
    """
    close = df["close"].values.astype(float)
    n = len(close)

    fwd = np.full((n, fold_half), np.nan)
    rev = np.full((n, fold_half), np.nan)

    for k in range(fold_half):
        fwd[fold_half:n - fold_half, k] = close[fold_half + k:n - fold_half + k]
        rev[fold_half:n - fold_half, k] = close[fold_half - k - 1:n - fold_half - k - 1]

    fwd_mean = np.nanmean(fwd, axis=1)
    rev_mean = np.nanmean(rev, axis=1)
    fwd_centered = fwd - fwd_mean[:, None]
    rev_centered = rev - rev_mean[:, None]

    cov = np.nanmean(fwd_centered * rev_centered, axis=1)
    fwd_std = np.nanstd(fwd, axis=1)
    rev_std = np.nanstd(rev, axis=1)

    denom = fwd_std * rev_std
    corr = np.where(denom > 1e-10, cov / denom, 0.0)

    score_series = pd.Series(corr, index=df.index)
    is_reversal = score_series < -min_correlation
    return is_reversal.shift(1).fillna(False)


def volume_filter(df: pd.DataFrame, multiplier: float = 1.2) -> pd.Series:
    avg_vol = df["volume"].rolling(20).mean()
    return (df["volume"] > avg_vol * multiplier).shift(1).fillna(False)


def adx_filter(df: pd.DataFrame, period: int = 14, threshold: float = 25) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return (adx > threshold).shift(1).fillna(False)


def time_filter(index: pd.DatetimeIndex) -> pd.Series:
    try:
        et = index.tz_convert("America/New_York")
    except TypeError:
        et = index
    is_open_window = (et.time >= pd.Timestamp("10:00").time()) & (
        et.time <= pd.Timestamp("15:30").time()
    )
    is_monday_first = (et.dayofweek == 0) & (et.hour < 10)
    return pd.Series(is_open_window & ~is_monday_first, index=index)


FILTERS = {
    "trend_filter": trend_filter,
    "volatility_regime": volatility_regime_filter,
    "rsi_filter": rsi_filter,
    "adaptive_rsi": adaptive_rsi_filter,
    "volume_filter": volume_filter,
    "adx_filter": adx_filter,
    "time_filter": time_filter,
    "convolution_reversal": convolution_reversal_filter,
    "sentiment_filter": sentiment_filter,
    "fundamentals_filter": fundamentals_filter,
    "insider_filter": insider_filter,
    "reflection_filter": reflection_filter,
}


def apply_filters(entries: pd.Series, df: pd.DataFrame, filters: list[str], symbol: str = "SPY") -> pd.Series:
    mask = pd.Series(True, index=entries.index)
    for name in filters:
        if name not in FILTERS:
            continue
        fn = FILTERS[name]
        if name == "time_filter":
            result = fn(df.index)
        elif name in ("trend_filter", "rsi_filter", "adaptive_rsi"):
            long_f, _ = fn(df)
            result = long_f
        elif name in ("sentiment_filter", "fundamentals_filter", "insider_filter"):
            result = fn(df, ticker=symbol)
        elif name == "reflection_filter":
            result = fn(df, symbol=symbol)
        else:
            result = fn(df)
        result = result.reindex(entries.index, fill_value=False)
        mask = mask & result
    return entries & mask
