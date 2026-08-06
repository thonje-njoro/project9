"""Parameter search spaces for each strategy."""

import numpy as np
import pandas as pd


MEAN_REVERSION_GRID = {
    "period": np.array([10, 15, 20, 25, 30]),
    "std_threshold": np.array([1.0, 1.5, 2.0, 2.5]),
    "atr_stop_mult": np.array([1.0, 1.5, 2.0, 2.5]),
}

MOMENTUM_GRID = {
    "breakout_period": np.array([10, 15, 20, 25, 30]),
    "volume_multiplier": np.array([1.0, 1.5, 2.0]),
    "trail_atr_mult": np.array([1.5, 2.0, 2.5, 3.0]),
    "min_breakout_atr": np.array([0.5, 1.0, 1.5]),
}

TREND_FOLLOWING_GRID = {
    "fast_ema": np.array([10, 20, 30]),
    "slow_ema": np.array([50, 100, 150, 200]),
    "trail_atr_mult": np.array([2.0, 2.5, 3.0, 3.5]),
    "entry_atr_buffer": np.array([0.0, 0.25, 0.5]),
}

GRIDS = {
    "mean_reversion": MEAN_REVERSION_GRID,
    "momentum_breakout": MOMENTUM_GRID,
    "trend_following": TREND_FOLLOWING_GRID,
}


def get_grid(strategy: str) -> dict:
    if strategy not in GRIDS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(GRIDS.keys())}")
    return GRIDS[strategy]


def grid_combinations(grid: dict) -> pd.DataFrame:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    mesh = np.meshgrid(*values, indexing="ij")
    flat = [m.ravel() for m in mesh]
    df = pd.DataFrame(dict(zip(keys, flat)))

    total = len(df)
    print(f"Total parameter combinations: {total:,}")
    return df
