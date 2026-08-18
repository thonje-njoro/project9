"""
reporting/metrics.py — Daily-return Sharpe, max drawdown, Calmar ratio.
CRITICAL: Do NOT use vectorbt's built-in Sharpe — it annualizes incorrectly
for intraday data. Always compute from daily returns.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_daily_returns(portfolio_returns: pd.Series) -> pd.Series:
    """
    Resample portfolio returns to daily frequency.

    Args:
        portfolio_returns: Bar-level return series.

    Returns:
        Daily return series.
    """
    if portfolio_returns.empty:
        return pd.Series(dtype=float)
    # If index is not DatetimeIndex, treat each observation as a daily return
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        return portfolio_returns.dropna()
    daily = portfolio_returns.resample("1D").sum().dropna()
    return daily


def compute_sharpe(daily_returns: pd.Series,
                   risk_free: float = 0.045) -> float:
    """
    Compute annualized Sharpe ratio from daily returns.

    Formula: (mean(daily_excess) / std(daily_excess)) × √252

    Args:
        daily_returns: Series of daily portfolio returns.
        risk_free: Annual risk-free rate.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(daily_returns) < 10:
        return 0.0

    excess = daily_returns - risk_free / 252
    if excess.std() == 0:
        return 0.0

    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown as a positive fraction.

    Args:
        equity_curve: Cumulative equity series.

    Returns:
        Maximum drawdown (0 to 1).
    """
    if equity_curve.empty:
        return 0.0

    running_max = equity_curve.cummax()
    drawdown = (running_max - equity_curve) / running_max
    return float(drawdown.max())


def compute_calmar(daily_returns: pd.Series,
                   equity_curve: pd.Series,
                   risk_free: float = 0.045) -> float:
    """
    Compute Calmar ratio = annualized return / max drawdown.

    Args:
        daily_returns: Daily return series.
        equity_curve: Cumulative equity series.
        risk_free: Annual risk-free rate.

    Returns:
        Calmar ratio.
    """
    if len(daily_returns) < 10:
        return 0.0

    annual_return = daily_returns.mean() * 252 - risk_free
    max_dd = compute_max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    return float(annual_return / max_dd)


def compute_sortino(daily_returns: pd.Series,
                    risk_free: float = 0.045) -> float:
    """
    Compute Sortino ratio (penalizes only downside volatility).

    Args:
        daily_returns: Daily return series.
        risk_free: Annual risk-free rate.

    Returns:
        Annualized Sortino ratio.
    """
    if len(daily_returns) < 10:
        return 0.0

    excess = daily_returns - risk_free / 252
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std() == 0:
        return 0.0 if excess.mean() <= 0 else np.inf

    return float(excess.mean() / downside.std() * np.sqrt(252))


def compute_trade_stats(trade_returns: np.ndarray) -> dict:
    """
    Compute win rate, average win/loss, profit factor.

    Args:
        trade_returns: Array of individual trade returns.

    Returns:
        dict with: n_trades, win_rate, avg_win, avg_loss, profit_factor
    """
    if len(trade_returns) == 0:
        return {
            "n_trades": 0, "win_rate": 0, "avg_win": 0,
            "avg_loss": 0, "profit_factor": 0,
        }

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "n_trades": int(len(trade_returns)),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
    }


def full_metrics(portfolio_returns: pd.Series,
                 equity_curve: pd.Series,
                 risk_free: float = 0.045) -> dict:
    """
    Compute all metrics for a backtest result.

    Args:
        portfolio_returns: Bar-level return series.
        equity_curve: Cumulative equity series.
        risk_free: Annual risk-free rate.

    Returns:
        dict with all computed metrics.
    """
    daily_returns = compute_daily_returns(portfolio_returns)

    return {
        "sharpe": compute_sharpe(daily_returns, risk_free),
        "sortino": compute_sortino(daily_returns, risk_free),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "calmar": compute_calmar(daily_returns, equity_curve, risk_free),
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0,
        "n_trading_days": len(daily_returns),
    }
