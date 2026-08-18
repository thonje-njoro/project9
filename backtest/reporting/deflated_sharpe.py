"""
reporting/deflated_sharpe.py — Deflated Sharpe Ratio from López de Prado.
'Advances in Financial ML', Chapter 8. Accounts for multiple testing inflation.
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def deflated_sharpe_ratio(max_sharpe: float, n_trials: int, n_obs: int,
                          skewness: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR).

    Accounts for multiple testing inflation of Sharpe Ratio.
    DSR > 0.95 means the observed Sharpe is in the 95th percentile
    of what we'd expect from random strategies.

    Args:
        max_sharpe: Best observed Sharpe ratio.
        n_trials: Number of strategies/trials tested.
        n_obs: Number of observations (trading days).
        skewness: Return distribution skewness.
        kurtosis: Return distribution kurtosis (excess + 3).

    Returns:
        DSR as a probability (0 to 1). Gate: must be > 0.95.
    """
    if n_obs < 10 or n_trials < 1:
        return 0.0

    # Expected max Sharpe under null (no skill)
    # Using Euler-Mascheroni constant for extreme value theory
    gamma = np.log(n_trials) + 0.5772  # Euler-Mascheroni
    sharpe_star = np.sqrt(2 * gamma) / np.sqrt(n_obs)

    # Adjust for non-normality
    sharpe_adj = sharpe_star * np.sqrt(
        1 - skewness * sharpe_star + ((kurtosis - 1) / 4) * sharpe_star**2
    )

    # Z-statistic for the observed Sharpe
    denominator = np.sqrt(
        1 - skewness * max_sharpe + ((kurtosis - 1) / 4) * max_sharpe**2
    )
    if denominator <= 0:
        return 0.0

    z = (max_sharpe - sharpe_adj) * np.sqrt(n_obs - 1) / denominator

    return float(stats.norm.cdf(z))


def compute_dsr_from_returns(returns: np.ndarray, n_trials: int = 10,
                             risk_free: float = 0.045) -> dict:
    """
    Compute DSR from a return series.

    Args:
        returns: Array of daily returns.
        n_trials: Number of strategy variants tested.
        risk_free: Annual risk-free rate.

    Returns:
        dict with: sharpe, dsr, skewness, kurtosis, n_obs, passes
    """
    if len(returns) < 10:
        return {"sharpe": 0, "dsr": 0, "passes": False}

    excess = returns - risk_free / 252
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    from scipy.stats import skew, kurtosis as scipy_kurtosis
    skew_val = float(skew(returns))
    kurt_val = float(scipy_kurtosis(returns, fisher=False))  # Non-excess kurtosis

    dsr = deflated_sharpe_ratio(
        max_sharpe=sharpe,
        n_trials=n_trials,
        n_obs=len(returns),
        skewness=skew_val,
        kurtosis=kurt_val,
    )

    return {
        "sharpe": sharpe,
        "dsr": dsr,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "n_obs": len(returns),
        "passes": dsr > 0.95,
    }
