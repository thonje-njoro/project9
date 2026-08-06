"""System health monitor with deployment-time kill-switches.

Implements kill-switch thresholds from aligrithm.com:
'How to Detect When a Trading System Is Dying'

Four hard thresholds written at deployment:
1. Drawdown trigger: >1.5x historical max drawdown
2. Sharpe trigger: <0.3 for 6 consecutive months
3. Profit factor trigger: <1.1 for 6 consecutive months
4. CUSUM trigger: h=5 on standardized expectancy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HealthStatus:
    healthy: bool = True
    breach_reason: str = ""
    breach_date: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class SystemHealthMonitor:
    """
    Monitors live/paper trading system health using hard kill-switch thresholds.

    These thresholds are set at deployment time and cannot be changed mid-run,
    preventing the common failure mode of loosening rules during drawdowns.
    """

    def __init__(
        self,
        historical_max_dd: float = 0.10,
        sharpe_trigger: float = 0.3,
        profit_factor_trigger: float = 1.1,
        lookback_months: int = 6,
        cusum_h: float = 5.0,
        cusum_target: float = 0.0,
    ):
        self.historical_max_dd = historical_max_dd
        self.sharpe_trigger = sharpe_trigger
        self.profit_factor_trigger = profit_factor_trigger
        self.lookback_months = lookback_months
        self.cusum_h = cusum_h
        self.cusum_target = cusum_target
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

    def check_drawdown(
        self, equity_curve: pd.Series, date: Optional[str] = None
    ) -> HealthStatus:
        """
        Kill-switch 1: Current drawdown exceeds 1.5x historical max.

        This catches regime shifts where the system is losing more than
        it ever did in backtest — a clear sign the model is broken.
        """
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        current_dd = abs(dd.iloc[-1])
        threshold = self.historical_max_dd * 1.5

        if current_dd > threshold:
            return HealthStatus(
                healthy=False,
                breach_reason=f"Drawdown {current_dd:.2%} exceeds 1.5x historical max ({threshold:.2%})",
                breach_date=date or str(equity_curve.index[-1]),
                metrics={"current_dd": current_dd, "threshold": threshold},
            )
        return HealthStatus(metrics={"current_dd": current_dd, "threshold": threshold})

    def check_sharpe(
        self, equity_curve: pd.Series, date: Optional[str] = None
    ) -> HealthStatus:
        """
        Kill-switch 2: Rolling Sharpe < 0.3 for lookback_months.

        A persistently low Sharpe indicates the strategy has stopped
        capturing alpha — the edge has degraded.
        """
        returns = equity_curve.pct_change().dropna()
        if len(returns) < 20:
            return HealthStatus()

        rolling_sharpe = returns.rolling(60).mean() / (returns.rolling(60).std() + 1e-10) * np.sqrt(252)
        recent = rolling_sharpe.iloc[-min(120, len(rolling_sharpe)):]

        below_trigger = (recent < self.sharpe_trigger).sum()
        months_available = len(recent) / 20

        if months_available >= self.lookback_months and below_trigger >= self.lookback_months * 15:
            return HealthStatus(
                healthy=False,
                breach_reason=f"Rolling Sharpe below {self.sharpe_trigger} for {self.lookback_months}+ months",
                breach_date=date or str(equity_curve.index[-1]),
                metrics={"rolling_sharpe_last": float(recent.iloc[-1]), "months_below": below_trigger / 15},
            )
        return HealthStatus(metrics={"rolling_sharpe_last": float(recent.iloc[-1])})

    def check_profit_factor(
        self, equity_curve: pd.Series, date: Optional[str] = None
    ) -> HealthStatus:
        """
        Kill-switch 3: Profit factor < 1.1 for lookback_months.

        A profit factor near 1.0 means gross profits barely exceed gross losses —
        the strategy is barely profitable after transaction costs.
        """
        returns = equity_curve.pct_change().dropna()
        if len(returns) < 20:
            return HealthStatus()

        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        pf = gains / (losses + 1e-10)

        if pf < self.profit_factor_trigger:
            return HealthStatus(
                healthy=False,
                breach_reason=f"Profit factor {pf:.2f} below trigger {self.profit_factor_trigger}",
                breach_date=date or str(equity_curve.index[-1]),
                metrics={"profit_factor": pf, "threshold": self.profit_factor_trigger},
            )
        return HealthStatus(metrics={"profit_factor": pf})

    def check_cusum(
        self, equity_curve: pd.Series, date: Optional[str] = None
    ) -> HealthStatus:
        """
        Kill-switch 4: CUSUM h=5 on standardized expectancy.

        CUSUM detects small persistent shifts in expected return.
        When cumulative sum exceeds the threshold, the system's
        expected value has shifted negatively.
        """
        returns = equity_curve.pct_change().dropna()
        if len(returns) < 30:
            return HealthStatus()

        mu = returns.mean()
        sigma = returns.std()
        if sigma < 1e-10:
            return HealthStatus()

        standardized = (returns - self.cusum_target) / sigma
        self._cusum_pos = max(0, self._cusum_pos + standardized.iloc[-1] - 0.5)
        self._cusum_neg = min(0, self._cusum_neg + standardized.iloc[-1] + 0.5)

        if self._cusum_pos > self.cusum_h or self._cusum_neg < -self.cusum_h:
            direction = "negative" if self._cusum_neg < -self.cusum_h else "positive"
            return HealthStatus(
                healthy=False,
                breach_reason=f"CUSUM ({direction}) exceeded threshold {self.cusum_h}",
                breach_date=date or str(equity_curve.index[-1]),
                metrics={"cusum_pos": self._cusum_pos, "cusum_neg": self._cusum_neg},
            )
        return HealthStatus(metrics={"cusum_pos": self._cusum_pos, "cusum_neg": self._cusum_neg})

    def full_check(self, equity_curve: pd.Series, date: Optional[str] = None) -> HealthStatus:
        """Run all four kill-switch checks. Returns first breach found."""
        for check_fn in [self.check_drawdown, self.check_sharpe, self.check_profit_factor, self.check_cusum]:
            result = check_fn(equity_curve, date)
            if not result.healthy:
                return result
        return HealthStatus(metrics={"all_checks": "passed"})
