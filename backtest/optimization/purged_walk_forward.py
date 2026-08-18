"""
optimization/purged_walk_forward.py — Purged k-fold Walk-Forward Validation.
6 folds, each fold = 6-month train / 1-month test with 20-bar embargo.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PurgedWalkForward:
    """
    Purged Walk-Forward Validation with embargo periods.

    Prevents data leakage by:
    1. Purging: removing training data that overlaps with test labels
    2. Embargo: adding a gap between train and test sets
    """

    def __init__(self, n_splits: int = 6, embargo: int = 20):
        """
        Args:
            n_splits: Number of CV folds.
            embargo: Number of bars to skip between train and test.
        """
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, df: pd.DataFrame):
        """
        Generate purged walk-forward splits.

        Each fold: 6 months train, 1 month test, with embargo gap.

        Yields:
            (train_idx, test_idx) as integer arrays.
        """
        n = len(df)
        if n < 100:
            logger.warning("Too few bars for walk-forward split")
            return

        # Split into roughly equal test periods
        test_size = n // (self.n_splits + 2)  # Leave room for training data
        if test_size < 20:
            test_size = 20

        for fold in range(self.n_splits):
            # Test period: sequential chunk
            test_start = n - (self.n_splits - fold) * test_size
            test_end = test_start + test_size

            if test_start < 0 or test_end > n:
                continue

            # Embargo gap
            train_end = test_start - self.embargo
            if train_end <= 0:
                continue

            # Training data: everything before embargo
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def validate(self, df: pd.DataFrame,
                 strategy_fn: Callable,
                 params: dict,
                 metric_fn: Callable | None = None) -> dict:
        """
        Run purged walk-forward validation on a strategy.

        Args:
            df: Full OHLCV DataFrame.
            strategy_fn: Function that generates signals given (df, **params).
            params: Strategy parameters.
            metric_fn: Function that computes Sharpe from returns. If None,
                       uses a simple mean/std calculation.

        Returns:
            dict with:
                fold_sharpes: List of OOS Sharpe per fold
                train_test_correlation: Pearson corr between IS and OOS Sharpe
                consistency_rate: Fraction of folds where OOS Sharpe > 0
                mean_oos_sharpe: Average OOS Sharpe
                passes: True if mean_oos_sharpe > 0.5 and consistency_rate > 0.6
        """
        if metric_fn is None:
            metric_fn = _default_sharpe

        fold_sharpes = []
        train_sharpes = []

        for train_idx, test_idx in self.split(df):
            # Train period
            df_train = df.iloc[train_idx]
            # Test period
            df_test = df.iloc[test_idx]

            if len(df_train) < 50 or len(df_test) < 10:
                continue

            try:
                # Generate signals on training data (in-sample)
                import inspect
                sig = inspect.signature(strategy_fn)
                valid_keys = set(sig.parameters.keys()) - {"df"}
                filtered_params = {k: v for k, v in params.items() if k in valid_keys}

                result_train = strategy_fn(df=df_train, **filtered_params)
                result_test = strategy_fn(df=df_test, **filtered_params)

                # Extract entry/exit signals
                if len(result_train) == 5:
                    entries_train, exits_train, _, _, _ = result_train
                    entries_test, exits_test, _, _, _ = result_test
                else:
                    entries_train, exits_train, _, _ = result_train
                    entries_test, exits_test, _, _ = result_test

                # Compute returns (simplified: use close-to-close)
                train_returns = _compute_returns_from_signals(df_train, entries_train, exits_train)
                test_returns = _compute_returns_from_signals(df_test, entries_test, exits_test)

                train_sharpe = metric_fn(train_returns)
                test_sharpe = metric_fn(test_returns)

                train_sharpes.append(train_sharpe)
                fold_sharpes.append(test_sharpe)

            except Exception as e:
                logger.warning(f"WFV fold failed: {e}")
                continue

        if not fold_sharpes:
            return {
                "fold_sharpes": [],
                "train_test_correlation": 0,
                "consistency_rate": 0,
                "mean_oos_sharpe": 0,
                "passes": False,
            }

        # Correlation between IS and OOS
        if len(train_sharpes) > 2:
            corr = np.corrcoef(train_sharpes, fold_sharpes)[0, 1]
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0

        consistency = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
        mean_oos = np.mean(fold_sharpes)

        passes = mean_oos > 0.5 and consistency > 0.6

        return {
            "fold_sharpes": fold_sharpes,
            "train_test_correlation": float(corr),
            "consistency_rate": float(consistency),
            "mean_oos_sharpe": float(mean_oos),
            "passes": passes,
        }


def _default_sharpe(returns: np.ndarray, risk_free: float = 0.045) -> float:
    """Default Sharpe calculation from daily returns."""
    if len(returns) < 5:
        return 0.0
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _compute_returns_from_signals(df: pd.DataFrame,
                                  entries: pd.Series,
                                  exits: pd.Series) -> np.ndarray:
    """
    Compute trade returns from entry/exit signals.
    Simplified: captures close-to-close returns between entry and exit.
    """
    returns = []
    in_position = False
    entry_price = 0.0

    for i in range(len(df)):
        if entries.iloc[i] and not in_position:
            in_position = True
            entry_price = df["close"].iloc[i]
        elif exits.iloc[i] and in_position:
            exit_price = df["close"].iloc[i]
            if entry_price > 0:
                returns.append((exit_price - entry_price) / entry_price)
            in_position = False

    return np.array(returns) if returns else np.array([0.0])
