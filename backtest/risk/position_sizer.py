"""ATR-based position sizing with Kelly-fraction scaling.

Adds Fractional Kelly position sizing (Item 1) alongside existing ATR sizing.
Kelly fraction scales the ATR-based size up/down based on each instrument's
historical edge, per Kelly Criterion theory.

Kelly logic:
  f* = (p * avg_win - q * avg_loss) / (avg_win * avg_loss)
     = p - q / R   where R = avg_win / avg_loss

We use Quarter-Kelly (0.25*f) to be conservative — full Kelly is too volatile.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def atr_position_sizes(
    equity: float,
    atr: pd.Series,
    price: pd.Series,
    risk_pct: float = 0.01,
    max_exposure_pct: float = 0.25,
    kelly_mult: float = 1.0,
) -> pd.Series:
    """Standard ATR-based position sizing, optionally scaled by Kelly multiplier.

    Args:
        equity: Account equity
        atr: ATR series
        price: Price series
        risk_pct: Base risk per trade
        max_exposure_pct: Max exposure cap as fraction of equity
        kelly_mult: Kelly fraction multiplier (1.0 = ATR-only, 0.25 = quarter-Kelly)

    Returns:
        Position sizes in units
    """
    risk_dollars = equity * risk_pct * kelly_mult
    atr_sizes = (risk_dollars / atr).clip(lower=0)

    notional_cap = equity * max_exposure_pct
    max_units = (notional_cap / price).clip(lower=0)

    sizes = pd.concat([atr_sizes, max_units], axis=1).min(axis=1)
    return sizes.fillna(0)


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Compute Fractional Kelly position size multiplier.

    Args:
        win_rate: Historical win rate (0.0-1.0)
        avg_win: Average winning trade return (positive, e.g. 0.02 for 2%)
        avg_loss: Average losing trade return (positive magnitude, e.g. 0.01)
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter-Kelly)

    Returns:
        Kelly multiplier (0.0 to max_kelly). 1.0 = normal ATR sizing.
        0.0 = skip this instrument (negative edge).
    """
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
        return 1.0  # fall back to ATR-only

    # Edge: expected return per trade
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss
    if edge <= 0:
        return 0.0  # negative edge — skip

    # Kelly formula: f* = edge / (avg_win * avg_loss)  [continuous approximation]
    # Or: f* = win_rate - (1-win_rate) / (avg_win/avg_loss)
    r = avg_win / avg_loss
    full_kelly = win_rate - (1 - win_rate) / r

    if full_kelly <= 0:
        return 0.0

    return min(full_kelly * kelly_fraction, 2.0)  # cap at 2x to avoid insanity


def compute_kelly_from_trades(trade_returns: pd.Series,
                               kelly_fraction: float = 0.25) -> Tuple[float, float, float, float]:
    """Compute Kelly metrics from actual trade return series.

    Args:
        trade_returns: Series of per-trade returns
        kelly_fraction: Fraction of full Kelly to use (default 0.25)

    Returns:
        (win_rate, avg_win, avg_loss, kelly_mult)
    """
    if len(trade_returns) < 5:
        return 0.0, 0.0, 0.0, 1.0

    winners = trade_returns[trade_returns > 0]
    losers = trade_returns[trade_returns <= 0]

    if len(winners) == 0 or len(losers) == 0:
        return 0.0, 0.0, 0.0, 1.0

    win_rate = len(winners) / len(trade_returns)
    avg_win = float(winners.mean())
    avg_loss = float(abs(losers.mean()))

    kelly = compute_kelly_fraction(win_rate, avg_win, avg_loss, kelly_fraction=kelly_fraction)
    return win_rate, avg_win, avg_loss, kelly


def progressive_atr_position_sizes(
    equity: float,
    atr: pd.Series,
    price: pd.Series,
    risk_pct: float = 0.01,
    max_exposure_pct: float = 0.25,
    consecutive_losses: int = 0,
    progressive_thresholds: Optional[list] = None,
) -> pd.Series:
    """ATR position sizing with progressive risk reduction.

    After N consecutive losses, risk is reduced by a factor.
    This extends survival time during drawdowns.

    Args:
        equity: Account equity
        atr: ATR series
        price: Price series
        risk_pct: Base risk per trade
        max_exposure_pct: Max exposure cap
        consecutive_losses: Number of consecutive losses so far
        progressive_thresholds: List of (losses, reduction_factor) tuples.

    Returns:
        Position sizes with risk reduction applied
    """
    if progressive_thresholds is None:
        progressive_thresholds = [(2, 0.75), (4, 0.5), (6, 0.25)]

    reduction = 1.0
    for loss_threshold, factor in progressive_thresholds:
        if consecutive_losses >= loss_threshold:
            reduction = factor

    adjusted_risk = risk_pct * reduction
    sizes = atr_position_sizes(equity, atr, price, adjusted_risk, max_exposure_pct)
    return sizes


def compute_max_position_size(
    equity: float,
    price: float,
    daily_loss_limit_pct: float = 0.05,
    max_trades_per_day: int = 3,
) -> float:
    """Compute maximum position size given daily loss limit.

    From prop firm rules: daily loss limit is 5%.
    If we allow max_trades_per_day, each trade can risk at most
    (daily_limit / max_trades) of the account.
    """
    risk_per_trade = equity * daily_loss_limit_pct / max(max_trades_per_day, 1)
    position = risk_per_trade / max(price, 0.01)
    return position


# ════════════════════════════════════════════════════════════════
# Prop Firm Challenge Position Sizing (separate sprint-mode)
# ════════════════════════════════════════════════════════════════

def challenge_position_sizes(
    equity: float,
    atr: pd.Series,
    price: pd.Series,
    risk_per_trade_pct: float = 0.0085,
    max_exposure_pct: float = 2.0,
    phase_multiplier: float = 1.0,
    consecutive_losses: int = 0,
    progressive_reduction: list = None,
    daily_loss_remaining_pct: float = 1.0,
) -> pd.Series:
    """Position sizing for prop firm challenge sprint (separate from slow system).

    Features:
      - Fixed fractional risk (0.85% of current equity per trade)
      - Phase-aware multiplier (probing=0.5, acceleration=1.0, preservation=0.35-0.7)
      - Progressive reduction on consecutive losses
      - Daily loss budget scaling (reduce size if near daily limit)
      - Higher max exposure (200% notional, intraday leverage is acceptable)

    Args:
        equity: Current account equity
        atr: ATR series for stop distance
        price: Price series
        risk_per_trade_pct: Base risk per trade (default: 0.0085 = 0.85%)
        max_exposure_pct: Max notional as fraction of equity (default: 2.0 = 200%)
        phase_multiplier: Lifecycle phase multiplier (0.5 probing, 1.0 accel, etc.)
        consecutive_losses: Current streak of consecutive losing trades
        progressive_reduction: [(loss_streak, factor), ...] applied to risk
        daily_loss_remaining_pct: Fraction of daily loss budget remaining (1.0=full)

    Returns:
        Position sizes in units (shares)
    """
    if progressive_reduction is None:
        progressive_reduction = [(2, 0.75), (3, 0.50), (4, 0.25)]

    # === Base risk in dollars ===
    risk_dollars = equity * risk_per_trade_pct * phase_multiplier

    # === Progressive reduction on consecutive losses ===
    reduction = 1.0
    for loss_threshold, factor in progressive_reduction:
        if consecutive_losses >= loss_threshold:
            reduction = factor
    risk_dollars *= reduction

    # === Daily loss budget scaling ===
    # If we've already used 50% of the daily loss budget,
    # cut remaining trade sizes by half
    risk_dollars *= daily_loss_remaining_pct

    # === ATR-based unit sizing ===
    # Position size = risk_dollars / (atr * stop_loss_r_mult)
    # Since atr already represents the expected stop distance
    atr_safe = atr.replace(0, float('nan')).ffill().fillna(1.0)
    atr_sizes = (risk_dollars / atr_safe).clip(lower=0)

    # === Notional cap (prevents over-concentration) ===
    notional_cap = equity * max_exposure_pct
    price_safe = price.replace(0, float('nan')).ffill().fillna(1.0)
    max_units = (notional_cap / price_safe).clip(lower=0)

    # === Final size = min(ATR-based, notional cap) ===
    sizes = pd.concat([atr_sizes, max_units], axis=1).min(axis=1)
    return sizes.fillna(0).astype(float)
