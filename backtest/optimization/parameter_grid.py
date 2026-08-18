"""
optimization/parameter_grid.py — Grid definitions per strategy for parameter sweep.
"""

# Parameter grids for sweep optimization
# Each entry: (param_name, min_value, max_value, step)
# Changes are limited to ±30% of current value per the AI loop rules.

PARAMETER_GRIDS = {
    "kalman_trend": {
        "Q": (0.005, 0.05, 0.005),
        "R": (0.5, 2.0, 0.25),
        "velocity_threshold_pct": (0.05, 0.20, 0.025),
        "trail_atr_mult": (1.5, 4.0, 0.5),
    },
    "cper_gld_ratio": {
        "z_entry": (1.0, 2.5, 0.2),
        "z_exit": (-0.5, 0.5, 0.25),
        "trail_atr_mult": (1.0, 3.0, 0.5),
        "window": (10, 30, 5),
    },
    "orb_strategy": {
        "orb_period": (1, 3, 1),
        "min_rel_volume": (0.5, 1.5, 0.1),
        "atr_stop_pct": (0.05, 0.20, 0.025),
        "atr_period": (10, 20, 2),
    },
    "momentum_orb": {
        "or_minutes": (5, 90, 5),
        "atr_mult_stop": (1.0, 3.0, 0.5),
        "trail_mult": (1.0, 3.0, 0.5),
        "min_volume": (50_000, 200_000, 25_000),
    },
    "vwap_mean_reversion": {
        "z_score_lookback": (10, 30, 5),
        "z_entry": (1.0, 3.0, 0.5),
        "adx_threshold": (20.0, 35.0, 5.0),
        "max_relative_volume": (1.0, 2.5, 0.25),
        "trail_atr_mult": (1.0, 3.0, 0.5),
    },
    "xauusd_session_mr": {
        "z_entry": (1.0, 2.5, 0.25),
        "z_exit": (-0.5, 0.5, 0.25),
        "trail_atr_mult": (1.0, 3.0, 0.5),
        "asian_range_multiplier": (1.5, 3.0, 0.25),
    },
}


def get_grid(strategy_name: str) -> dict:
    """Get the parameter grid for a strategy."""
    return PARAMETER_GRIDS.get(strategy_name, {})


def generate_sweep_params(strategy_name: str, current_params: dict,
                          max_change_pct: float = 0.30) -> list[dict]:
    """
    Generate parameter variants for sweep.
    Each variant changes ONE parameter by ±max_change_pct.

    Args:
        strategy_name: Strategy name.
        current_params: Current parameter values.
        max_change_pct: Max change per parameter (0.30 = 30%).

    Returns:
        List of parameter dicts (one change each).
    """
    grid = get_grid(strategy_name)
    if not grid:
        return []

    variants = []
    for param_name, (pmin, pmax, pstep) in grid.items():
        if param_name not in current_params:
            continue

        current = current_params[param_name]

        # Generate steps within ±30% of current
        low = max(pmin, current * (1 - max_change_pct))
        high = min(pmax, current * (1 + max_change_pct))

        value = low
        while value <= high:
            if abs(value - current) > 1e-6:  # Skip current value
                variant = current_params.copy()
                variant[param_name] = type(current)(value) if isinstance(current, int) else value
                variants.append(variant)
            value += pstep

    return variants
