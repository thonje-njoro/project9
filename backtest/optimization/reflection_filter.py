"""Reflection-based entry filter.

Uses lessons from past trades to block entries matching known losing patterns.
"""

import numpy as np
import pandas as pd
from typing import Optional


def reflection_filter(
    df: pd.DataFrame,
    symbol: str = "SPY",
    regime: Optional[str] = None,
    min_win_rate: float = 0.3,
    block_consecutive_losses: int = 3,
) -> pd.Series:
    """Block entries based on historical trade lessons.

    Args:
        df: OHLCV DataFrame.
        symbol: Ticker symbol.
        regime: Current regime label.
        min_win_rate: Minimum acceptable win rate to allow trading.
        block_consecutive_losses: Skip signal after N consecutive losses.

    Returns:
        Boolean Series (True = allow entry).
    """
    try:
        from risk.reflection import TradeReflector
        reflector = TradeReflector()
        lessons = reflector.get_lessons(symbol)
    except Exception:
        return pd.Series(True, index=df.index)

    allow = pd.Series(True, index=df.index)

    for lesson in lessons:
        if lesson.get("type") == "low_win_rate" and lesson.get("severity") == "high":
            allow = allow & False

        if lesson.get("type") == "regime_loss" and regime:
            if lesson.get("regime") == regime:
                allow = allow & False

        if lesson.get("type") == "consecutive_losses":
            trades = reflector.get_symbol_trades(symbol)[-block_consecutive_losses:]
            if len(trades) >= block_consecutive_losses:
                recent_returns = [t["return_pct"] for t in trades]
                if all(r < -2.0 for r in recent_returns):
                    allow = allow & False

    return allow.shift(1).fillna(True)
