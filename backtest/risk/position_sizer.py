"""
risk/position_sizer.py — ATR-based position sizing with 25% max exposure cap.
"""

import logging

import numpy as np

from backtest.config import RISK_CONFIG

logger = logging.getLogger(__name__)


def atr_position_size(equity: float, atr: float, price: float,
                      risk_pct: float | None = None,
                      max_exposure_pct: float | None = None) -> float:
    """
    Compute position size in shares based on ATR risk.

    Dollar risk per trade = equity × risk_pct
    Position size = dollar_risk / (atr × price) → in shares
    Capped at max_exposure_pct × equity / price

    CRITICAL: The 25% cap prevents >100% equity drawdowns on low-ATR instruments.
    This was a confirmed bug in the prior system.

    Args:
        equity: Current account equity in dollars.
        atr: Current ATR value (in price units).
        price: Current price of the instrument.
        risk_pct: Fraction of equity to risk per trade. Default from config.
        max_exposure_pct: Max position value as fraction of equity. Default 0.25.

    Returns:
        Number of shares (float). May be fractional for forex.
    """
    if risk_pct is None:
        risk_pct = RISK_CONFIG["risk_per_trade"]
    if max_exposure_pct is None:
        max_exposure_pct = RISK_CONFIG["max_exposure_pct"]

    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0

    dollar_risk = equity * risk_pct
    shares = dollar_risk / (atr * price)
    max_shares = (equity * max_exposure_pct) / price

    result = min(shares, max_shares)

    if shares > max_shares:
        logger.debug(
            f"ATR sizing capped: raw={shares:.2f}, cap={max_shares:.2f} "
            f"({max_exposure_pct*100:.0f}% of equity)"
        )

    return result


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   cap: float = 0.25) -> float:
    """
    Compute Kelly criterion fraction, capped.

    Args:
        win_rate: Fraction of winning trades (0 to 1).
        avg_win: Average winning trade return (positive).
        avg_loss: Average losing trade return (positive number).
        cap: Maximum Kelly fraction to allow.

    Returns:
        Optimal bet fraction, capped at `cap`.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    b = avg_win / avg_loss  # Win/loss ratio
    kelly = (win_rate * b - (1 - win_rate)) / b

    # Half-Kelly is more conservative and practical
    kelly = kelly / 2.0

    return max(0.0, min(kelly, cap))
