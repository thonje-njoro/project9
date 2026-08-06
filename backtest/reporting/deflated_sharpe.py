"""Deflated Sharpe Ratio — adjusts Sharpe significance for multiple testing.

From arXiv:2512.12924 and Marcos López de Prado's 'Advances in Financial ML':
The standard Sharpe Ratio doesn't account for the number of trials run.
With 80 parameter combinations tested, the critical Sharpe threshold is
NOT 0.0 — it's approximately sqrt(2 * ln(N_trials)) / sqrt(T_obs).

Usage:
    from reporting.deflated_sharpe import deflated_sharpe_ratio, dsr_critical
    dsr = deflated_sharpe_ratio(max_sharpe=1.5, n_trials=80, n_obs=1000)
    critical = dsr_critical(n_trials=80, n_obs=1000)
    print(f"DSR: {dsr:.2f}  (critical: {critical:.2f})")
"""

import numpy as np
from scipy import stats
from typing import Optional


def euler_mascheroni(n: int) -> float:
    """Euler-Mascheroni gamma approximation for large n."""
    return np.log(n) + 0.5772156649


def dsr_critical(
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Compute the critical Sharpe Ratio threshold for multiple testing.

    The DSR adjusts the Sharpe Ratio significance threshold based on the
    number of independent trials (parameter combinations tested) and the
    number of observations.

    Args:
        n_trials: Number of independent trials/parameter sets tested
        n_obs: Number of observations in the sample
        skewness: Return skewness (default 0 = normal)
        kurtosis: Return kurtosis (default 3 = normal)
        confidence: Statistical confidence level (default 0.95)

    Returns:
        Minimum Sharpe Ratio required for statistical significance
    """
    if n_trials <= 1 or n_obs < 10:
        return 0.0

    # Standard error of Sharpe Ratio (adjusted for non-normality)
    se_sharpe = np.sqrt((1 + 0.5 * skewness**2 - 0.75 * (kurtosis - 3)) / max(n_obs - 1, 1))

    # Expected maximum of N i.i.d. normal variables (E[max Z])
    e_max_z = np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
    e_max_z = max(e_max_z, 0)

    # Multiple testing correction: adjust for number of trials
    z_score = stats.norm.ppf(confidence)
    adjusted_threshold = e_max_z * se_sharpe + z_score * se_sharpe / np.sqrt(n_trials)

    return max(adjusted_threshold, 0.0)


def deflated_sharpe_ratio(
    max_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio.

    The DSR tells you the probability that the best backtest Sharpe is
    statistically significant after accounting for multiple testing.

    Args:
        max_sharpe: The maximum Sharpe observed across all trials
        n_trials: Number of independent trials tested
        n_obs: Number of observations
        skewness: Return skewness
        kurtosis: Return kurtosis

    Returns:
        DSR value. DSR > 0 means the Sharpe is statistically significant.
    """
    if n_trials <= 1 or n_obs < 10:
        return max_sharpe

    critical = dsr_critical(n_trials, n_obs, skewness, kurtosis)
    return max_sharpe - critical


def min_significant_sharpe(
    n_trials: int,
    n_obs: int,
    confidence: float = 0.95,
) -> float:
    """Quick helper: what Sharpe do I need for this to be real?

    Example:
        >>> min_significant_sharpe(80, 1000)
        0.29  # Sharpe must be > 0.29 to be significant with 80 trials
    """
    return dsr_critical(n_trials, n_obs, confidence=confidence)
