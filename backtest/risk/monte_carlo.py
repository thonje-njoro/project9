"""
risk/monte_carlo.py — Bootstrap Monte Carlo simulation (2,000 resamplings).
"""

import logging

import numpy as np
import pandas as pd

from backtest.config import VALIDATION_THRESHOLDS

logger = logging.getLogger(__name__)


def bootstrap_sharpe(returns: np.ndarray, n_simulations: int = 2000,
                     seed: int = 42) -> np.ndarray:
    """
    Bootstrap Sharpe ratios by resampling trades with replacement.

    Args:
        returns: Array of individual trade returns.
        n_simulations: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Array of Sharpe ratios from each simulation.
    """
    rng = np.random.RandomState(seed)
    n_trades = len(returns)
    sharpes = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Resample with replacement
        sample = rng.choice(returns, size=n_trades, replace=True)
        if sample.std() > 0:
            sharpes[i] = sample.mean() / sample.std() * np.sqrt(252)
        else:
            sharpes[i] = 0.0

    return sharpes


def bootstrap_drawdown(returns: np.ndarray, n_simulations: int = 2000,
                       seed: int = 42) -> np.ndarray:
    """
    Bootstrap maximum drawdowns by resampling trades with replacement.

    Args:
        returns: Array of individual trade returns.
        n_simulations: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Array of max drawdowns (as positive fractions) from each simulation.
    """
    rng = np.random.RandomState(seed)
    n_trades = len(returns)
    max_dds = np.zeros(n_simulations)

    for i in range(n_simulations):
        sample = rng.choice(returns, size=n_trades, replace=True)
        equity_curve = np.cumprod(1 + sample)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dds[i] = drawdowns.max()

    return max_dds


def monte_carlo_analysis(trade_returns: np.ndarray,
                         n_simulations: int | None = None) -> dict:
    """
    Full Monte Carlo analysis: Sharpe and drawdown distributions.

    Args:
        trade_returns: Array of individual trade returns.
        n_simulations: Number of bootstrap iterations. Default from config.

    Returns:
        dict with keys:
            sharpe_median, sharpe_5pct, sharpe_95pct,
            dd_median, dd_95pct, dd_99pct,
            survival_rate (fraction with DD < kill threshold),
            passes (bool: True if dd_99pct < kill threshold)
    """
    if n_simulations is None:
        n_simulations = VALIDATION_THRESHOLDS["mc_simulations"]

    if len(trade_returns) < 10:
        logger.warning(f"Too few trades ({len(trade_returns)}) for Monte Carlo")
        return {
            "sharpe_median": 0, "sharpe_5pct": 0, "sharpe_95pct": 0,
            "dd_median": 0, "dd_95pct": 0, "dd_99pct": 1.0,
            "survival_rate": 0, "passes": False,
        }

    returns = np.asarray(trade_returns, dtype=np.float64)

    sharpes = bootstrap_sharpe(returns, n_simulations)
    max_dds = bootstrap_drawdown(returns, n_simulations)

    kill_threshold = VALIDATION_THRESHOLDS["mc_max_dd_kill"]
    survival_rate = float((max_dds < kill_threshold).mean())

    result = {
        "sharpe_median": float(np.median(sharpes)),
        "sharpe_5pct": float(np.percentile(sharpes, 5)),
        "sharpe_95pct": float(np.percentile(sharpes, 95)),
        "dd_median": float(np.median(max_dds)),
        "dd_95pct": float(np.percentile(max_dds, 95)),
        "dd_99pct": float(np.percentile(max_dds, 99)),
        "survival_rate": survival_rate,
        "passes": survival_rate >= VALIDATION_THRESHOLDS["min_mc_survival_rate"],
    }

    logger.info(
        f"Monte Carlo ({n_simulations} sims): "
        f"Sharpe median={result['sharpe_median']:.2f}, "
        f"DD 99th%={result['dd_99pct']:.3f}, "
        f"Survival={survival_rate:.1%}"
    )

    return result
