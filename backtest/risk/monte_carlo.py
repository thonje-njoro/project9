"""Monte Carlo trade bootstrap for robustness validation.

From aligrithm.com 3.21:
Monte Carlo has two flavors: bootstrap of trades (IS-distribution) and
synthetic paths (model-conditional). Set kill-switches at bootstrap 99th
percentile, not IS maximum. Not a substitute for OOS.

Usage:
    from risk.monte_carlo import MonteCarloBootstrap
    mc = MonteCarloBootstrap(portfolio, n_simulations=1000)
    result = mc.run()
    print(f"99th percentile drawdown: {result['dd_99pct']:.2%}")
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCResult:
    """Monte Carlo simulation results."""
    n_simulations: int
    sharpe_99pct: float
    sharpe_95pct: float
    sharpe_median: float
    dd_99pct: float
    dd_95pct: float
    dd_median: float
    return_99pct: float
    return_95pct: float
    return_median: float
    return_01pct: float  # worst-case return
    max_consecutive_losses_99pct: int
    survival_rate: float  # pct of sims that didn't hit max DD
    all_sharpes: list = field(default_factory=list)
    all_drawdowns: list = field(default_factory=list)
    all_returns: list = field(default_factory=list)


class MonteCarloBootstrap:
    """Bootstrap portfolio trades via resampling to estimate outcome distribution."""

    def __init__(
        self,
        portfolio: vbt.Portfolio,
        n_simulations: int = 1000,
        max_drawdown_threshold: float = 0.10,
        random_seed: int = 42,
    ):
        self.portfolio = portfolio
        self.n_simulations = n_simulations
        self.max_dd_threshold = max_drawdown_threshold
        self.rng = np.random.default_rng(random_seed)

    def run(self) -> MCResult:
        """Run Monte Carlo simulation by resampling trades with replacement."""
        trades = self.portfolio.trades
        n_trades = trades.count()

        if n_trades < 10:
            return MCResult(
                n_simulations=0,
                sharpe_99pct=0, sharpe_95pct=0, sharpe_median=0,
                dd_99pct=0, dd_95pct=0, dd_median=0,
                return_99pct=0, return_95pct=0, return_median=0,
                return_01pct=0, max_consecutive_losses_99pct=0,
                survival_rate=0,
            )

        trade_returns = np.array(trades.pnl.values)
        trade_durations = np.array(
            (trades.records_readable["Exit Timestamp"] - trades.records_readable["Entry Timestamp"])
            .dt.total_seconds().values
        ) if hasattr(trades, 'records_readable') else np.ones(n_trades) * 3600

        all_sharpes = []
        all_drawdowns = []
        all_returns = []
        all_consec_losses = []
        survived = 0

        for _ in range(self.n_simulations):
            sampled_returns = self.rng.choice(trade_returns, size=n_trades, replace=True)
            sampled_durations = self.rng.choice(trade_durations, size=n_trades, replace=True)

            cumulative = np.cumsum(sampled_returns)
            total_return = cumulative[-1] if len(cumulative) > 0 else 0

            peak = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else np.array([0])
            dd = (cumulative - peak) / (peak + 1e-10)
            max_dd = abs(np.min(dd)) if len(dd) > 0 else 0

            # Count consecutive losses
            losses = (sampled_returns < 0).astype(int)
            consec = 0
            max_consec = 0
            for l in losses:
                consec = consec + 1 if l else 0
                max_consec = max(max_consec, consec)

            initial_equity = getattr(self.portfolio, 'init_cash', 10_000)
            equity_series = initial_equity + cumulative
            daily_ret = pd.Series(equity_series).pct_change().dropna()
            sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-10) * np.sqrt(252)) if len(daily_ret) > 5 else 0

            all_sharpes.append(sharpe)
            all_drawdowns.append(max_dd)
            all_returns.append(total_return / initial_equity)
            all_consec_losses.append(max_consec)

            if max_dd < self.max_dd_threshold:
                survived += 1

        all_sharpes = np.array(all_sharpes)
        all_drawdowns = np.array(all_drawdowns)
        all_returns = np.array(all_returns)
        all_consec_losses = np.array(all_consec_losses)

        return MCResult(
            n_simulations=self.n_simulations,
            sharpe_99pct=float(np.percentile(all_sharpes, 99)),
            sharpe_95pct=float(np.percentile(all_sharpes, 95)),
            sharpe_median=float(np.median(all_sharpes)),
            dd_99pct=float(np.percentile(all_drawdowns, 99)),
            dd_95pct=float(np.percentile(all_drawdowns, 95)),
            dd_median=float(np.median(all_drawdowns)),
            return_99pct=float(np.percentile(all_returns, 99)),
            return_95pct=float(np.percentile(all_returns, 95)),
            return_median=float(np.median(all_returns)),
            return_01pct=float(np.percentile(all_returns, 1)),
            max_consecutive_losses_99pct=int(np.percentile(all_consec_losses, 99)),
            survival_rate=survived / self.n_simulations,
            all_sharpes=all_sharpes.tolist(),
            all_drawdowns=all_drawdowns.tolist(),
            all_returns=all_returns.tolist(),
        )

    def print_report(self, result: MCResult) -> None:
        """Print a formatted Monte Carlo report."""
        print("=" * 60)
        print("MONTE CARLO BOOTSTRAP REPORT")
        print("=" * 60)
        print(f"  Simulations:     {result.n_simulations}")
        print(f"  Survival rate:   {result.survival_rate:.1%} (below {self.max_dd_threshold:.0%} DD)")
        print()
        print("  Sharpe Ratio:")
        print(f"    99th percentile: {result.sharpe_99pct:.2f}")
        print(f"    95th percentile: {result.sharpe_95pct:.2f}")
        print(f"    Median:          {result.sharpe_median:.2f}")
        print()
        print("  Max Drawdown:")
        print(f"    99th percentile: {result.dd_99pct:.2%}")
        print(f"    95th percentile: {result.dd_95pct:.2%}")
        print(f"    Median:          {result.dd_median:.2%}")
        print()
        print("  Return:")
        print(f"    99th percentile: {result.return_99pct:.2%}")
        print(f"    95th percentile: {result.return_95pct:.2%}")
        print(f"    Median:          {result.return_median:.2%}")
        print(f"    1st percentile:  {result.return_01pct:.2%}")
        print()
        print(f"  Max Consecutive Losses (99th): {result.max_consecutive_losses_99pct}")
        print("=" * 60)
