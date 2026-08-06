"""Build minimal context for LLM signal validation.

Constructs a compact context string (~500 tokens) for the agent validator.
"""

import pandas as pd
import numpy as np


def build_context(
    symbol: str,
    df: pd.DataFrame,
    regime: str,
    action: str,
    price: float,
    atr: float = 0.0,
) -> str:
    """Build minimal context for LLM validation.

    Args:
        symbol: Ticker symbol.
        df: Recent OHLCV DataFrame (last 20+ bars).
        regime: Current regime label.
        action: Proposed action (long_entry, long_exit, etc.).
        price: Current price.
        atr: Current ATR value.

    Returns:
        Context string (~500 tokens max).
    """
    close = df["close"]
    returns = close.pct_change().dropna()

    last_5 = close.tail(5).values
    price_change_5 = (last_5[-1] / last_5[0] - 1) * 100 if len(last_5) >= 5 else 0

    last_20 = close.tail(20).values
    price_change_20 = (last_20[-1] / last_20[0] - 1) * 100 if len(last_20) >= 20 else 0

    vol_20 = float(returns.tail(20).std()) * np.sqrt(252) * 100 if len(returns) >= 20 else 0

    sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price

    high_20 = float(df["high"].tail(20).max()) if len(df) >= 20 else price
    low_20 = float(df["low"].tail(20).min()) if len(df) >= 20 else price

    context = f"""Trading signal validation for {symbol}:
Price: ${price:.2f} | ATR: ${atr:.2f}
Action: {action} | Regime: {regime}
5-bar change: {price_change_5:+.2f}% | 20-bar change: {price_change_20:+.2f}%
20-day vol: {vol_20:.1f}% | 20-day high: ${high_20:.2f} | 20-day low: ${low_20:.2f}
SMA20: ${sma_20:.2f} | SMA50: ${sma_50:.2f}
Position vs SMA20: {'above' if price > sma_20 else 'below'}
Position vs SMA50: {'above' if price > sma_50 else 'below'}"""

    return context
