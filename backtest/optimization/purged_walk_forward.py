"""Purged k-fold walk-forward validation with embargo periods.

From arXiv:2512.12924 (Interpretable Hypothesis-Driven Trading) and
Marcos López de Prado's Combinatorial Purged Cross-Validation (CPCV):

Standard walk-forward leaks information between train and test sets.
Purged k-fold creates embargo periods where test data near the train/test
boundary is excluded to prevent leakage.

Usage:
    from optimization.purged_walk_forward import PurgedWalkForward
    wf = PurgedWalkForward(df, strategy='mean_reversion', symbol='SPY',
                           n_splits=6, embargo=20)
    result = wf.validate(params={'period': 20, 'std_threshold': 1.5})
"""

import numpy as np
import pandas as pd
from typing import Optional


class PurgedWalkForward:
    """Walk-forward validation with purged (embargo) periods between folds.

    Unlike standard walk-forward which can leak information through:
    1. Overlapping train/test features (e.g., lagged returns)
    2. Serial correlation in performance metrics
    Purged walk-forward removes test observations near train boundaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: str,
        symbol: str,
        n_splits: int = 6,
        embargo: int = 20,
        min_train_pct: float = 0.4,
        random_seed: int = 42,
    ):
        self.df = df
        self.strategy = strategy
        self.symbol = symbol
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train_pct = min_train_pct
        self.rng = np.random.default_rng(random_seed)

    def _get_fold_indices(self) -> list[dict]:
        """Generate purged train/test split indices.

        Each fold has:
        - train: contiguous block of data
        - test: next contiguous block
        - purged_test: test minus embargo bars from train boundary
        """
        n = len(self.df)
        min_train = int(n * self.min_train_pct)
        test_size = (n - min_train) // max(self.n_splits - 1, 1)

        folds = []
        train_end = min_train

        for i in range(self.n_splits - 1):
            train_start = 0
            test_start = train_end
            test_end = min(test_start + test_size, n)

            # Purge: remove embargo bars from start of test set
            purge_start = test_start
            purge_end = min(test_start + self.embargo, test_end)

            folds.append({
                "fold": i,
                "train": list(range(train_start, train_end)),
                "test": list(range(test_start, test_end)),
                "purged_test_start": purge_end,
                "test_purged": list(range(purge_end, test_end)),
            })

            train_end = test_end

        # Last fold
        if train_end < n:
            folds.append({
                "fold": self.n_splits - 1,
                "train": list(range(0, train_end)),
                "test": list(range(train_end, n)),
                "purged_test_start": train_end + self.embargo,
                "test_purged": list(range(min(train_end + self.embargo, n), n)),
            })

        return folds

    def _compute_sharpe(self, equity: np.ndarray) -> float:
        """Compute annualized Sharpe from equity curve."""
        if len(equity) < 10:
            return 0.0
        daily = pd.Series(equity).resample("1D").last() if hasattr(pd.Series(equity), 'resample') else pd.Series(equity)
        returns = daily.pct_change().dropna()
        if len(returns) < 5 or returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    def validate(self, params: dict) -> dict:
        """Run purged walk-forward validation for given parameter set."""
        from engine import BacktestEngine

        # Build a partial config for this single instrument
        config = {
            "initial_capital": 10_000,
            "commission": 0.0005,
            "risk_free_rate": 0.045,
        }

        folds = self._get_fold_indices()
        fold_results = []

        for fold in folds:
            if len(fold["train"]) < 100:
                continue

            train_df = self.df.iloc[fold["train"]]
            test_df = self.df.iloc[fold["test_purged"]] if fold.get("test_purged") else self.df.iloc[fold["test"]]

            if len(test_df) < 20:
                continue

            # Run engine on train data
            engine = BacktestEngine(
                {self.symbol: train_df},
                config,
                use_regime_filter=False,
            )

            try:
                train_portfolios = engine.run()
                if self.symbol not in train_portfolios:
                    continue
                train_pf = train_portfolios[self.symbol]
                train_sharpe = self._compute_sharpe(train_pf.value().values)
            except Exception:
                continue

            # Run engine on test data
            engine_test = BacktestEngine(
                {self.symbol: test_df},
                config,
                use_regime_filter=False,
            )

            try:
                test_portfolios = engine_test.run()
                if self.symbol not in test_portfolios:
                    continue
                test_pf = test_portfolios[self.symbol]
                test_sharpe = self._compute_sharpe(test_pf.value().values)
            except Exception:
                continue

            fold_results.append({
                "fold": fold["fold"],
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "sharpe_decay": train_sharpe - test_sharpe if train_sharpe != 0 else 0,
            })

        if not fold_results:
            return {"oos_sharpe": 0, "recommendation": "INSUFFICIENT_DATA", "folds": []}

        oos_sharpes = [f["test_sharpe"] for f in fold_results]
        is_sharpes = [f["train_sharpe"] for f in fold_results]

        avg_oos = float(np.mean(oos_sharpes))
        avg_is = float(np.mean(is_sharpes))
        decay = avg_is - avg_oos

        # Detection rules from arXiv:2512.12924
        positive_windows = sum(1 for s in oos_sharpes if s > 0)
        decay_windows = sum(1 for f in fold_results if f["sharpe_decay"] > 0.5)

        if avg_oos < 0:
            recommendation = "REJECT"
        elif decay > 1.0:
            recommendation = "OVERFIT"
        elif positive_windows < len(fold_results) * 0.5:
            recommendation = "UNSTABLE"
        elif decay > 0.5:
            recommendation = "DEGRADING"
        else:
            recommendation = "PASS"

        return {
            "oos_sharpe": avg_oos,
            "is_sharpe": avg_is,
            "sharpe_decay": decay,
            "recommendation": recommendation,
            "positive_windows": positive_windows,
            "total_windows": len(fold_results),
            "folds": fold_results,
        }


def detect_overfitting(wf_result: dict) -> dict:
    """Detect overfitting from walk-forward results.

    Returns dict with overfitting_detected bool and list of warnings.
    """
    warnings = []
    oos = wf_result.get("oos_sharpe", 0)
    is_s = wf_result.get("is_sharpe", 0)
    decay = wf_result.get("sharpe_decay", 0)
    positive = wf_result.get("positive_windows", 0)
    total = wf_result.get("total_windows", 1)

    if oos <= 0:
        warnings.append(f"OOS Sharpe not positive: {oos:.2f}")

    if decay > 1.0:
        warnings.append(f"Sharpe decay > 1.0: IS={is_s:.2f} → OOS={oos:.2f}")

    if positive < total * 0.5:
        warnings.append(f"OOS Sharpe positive in only {positive}/{total} windows")

    if decay > 0.5 and oos < 0.5:
        warnings.append(f"High decay ({decay:.2f}) with weak OOS ({oos:.2f}): overfit likely")

    return {
        "overfitting_detected": len(warnings) > 0,
        "warnings": warnings,
    }
