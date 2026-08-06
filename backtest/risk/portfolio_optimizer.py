"""Risk Parity portfolio allocation (Item 2).

Replaces equal-weight allocation with Equal Risk Contribution (ERC).
Each instrument contributes equally to portfolio variance.

From Maillard et al. (2010): Risk parity portfolios consistently
outperform equal-weight and mean-variance out-of-sample.

Also provides:
- Correlation matrix visualization data
- Risk contribution breakdown
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import minimize


def risk_parity_weights(
    cov_matrix: np.ndarray,
    max_weight: float = 0.40,
    min_weight: float = 0.02,
) -> np.ndarray:
    """Find weights where each asset contributes equally to portfolio variance.

    Minimizes the sum of squared differences between each asset's risk
    contribution and the equal-target risk contribution.

    Args:
        cov_matrix: NxN covariance matrix of asset returns
        max_weight: Maximum weight per asset (cap concentration)
        min_weight: Minimum weight per asset (floor diversification)

    Returns:
        Array of weights summing to 1.0
    """
    n = cov_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    def risk_contributions(w):
        """Compute each asset's contribution to portfolio variance."""
        port_var = w @ cov_matrix @ w
        if port_var <= 0:
            return np.ones(n) / n
        mrc = cov_matrix @ w  # marginal risk contribution
        return w * mrc / np.sqrt(port_var)  # risk contribution

    def objective(w):
        rc = risk_contributions(w)
        target = rc.sum() / n
        return np.sum((rc - target)**2)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    bounds = [(min_weight, max_weight)] * n

    # Multiple starting points to avoid local minima
    best_result = None
    best_val = np.inf

    for seed_w in [
        np.ones(n) / n,
        # Volatility-weighted start (inverse vol)
        np.diag(cov_matrix) if n > 0 else np.ones(n) / n,
    ]:
        sv = np.array(seed_w).flatten()
        sv = sv / sv.sum()
        if len(sv) != n:
            continue

        result = minimize(
            objective,
            sv,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    if best_result is None or not best_result.success:
        # Fallback: inverse volatility weighting (simpler risk parity)
        inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / inv_vol.sum()
        return np.clip(weights, min_weight, max_weight)

    weights = best_result.x
    return np.clip(weights, min_weight, max_weight)


def compute_risk_parity_allocation(
    returns: pd.DataFrame,
    symbols: list[str],
    lookback: int = 60,
    max_weight: float = 0.40,
    min_weight: float = 0.02,
) -> pd.Series:
    """Compute risk parity weights from historical returns.

    Args:
        returns: DataFrame of daily returns, columns = symbols
        symbols: List of symbols to include
        lookback: Rolling window for covariance estimation
        max_weight: Max allocation per instrument
        min_weight: Min allocation per instrument

    Returns:
        Series with symbol -> weight mapping
    """
    if len(symbols) == 0:
        return pd.Series(dtype=float)

    # Use recent returns for covariance
    recent = returns[symbols].dropna().tail(lookback)
    if len(recent) < 10:
        return pd.Series(1.0 / len(symbols), index=symbols)

    # Shrinkage covariance for numerical stability
    cov = recent.cov()
    # Ledoit-Wolf style shrinkage toward diagonal
    shrunk_cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))

    try:
        weights = risk_parity_weights(
            shrunk_cov.values,
            max_weight=max_weight,
            min_weight=min_weight,
        )
        return pd.Series(weights, index=symbols)
    except Exception:
        # Fallback to equal weight
        return pd.Series(1.0 / len(symbols), index=symbols)


def compute_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> pd.Series:
    """Compute each asset's percentage contribution to portfolio risk.

    Returns:
        Series of risk contribution percentages (sums to 100%)
    """
    port_var = weights @ cov_matrix @ weights
    if port_var <= 0:
        return pd.Series(np.ones(len(weights)) / len(weights))

    mrc = cov_matrix @ weights
    rc = weights * mrc / np.sqrt(port_var)
    return pd.Series(rc / rc.sum())
