"""Degrees-of-freedom audit for trading systems.

From aligrithm.com 3.16:
Every tunable element in a trading system consumes a degree of freedom.
Parameters, filter thresholds, stop types, exit rules — all count.
High degrees of freedom with limited data = overfitting machine.

This module counts every degree of freedom across strategies, filters,
risk management, and exit rules, then computes the search-width bias.

Usage:
    from risk.dof_audit import DOFAudit
    auditor = DOFAudit(strategy='mean_reversion', params={...})
    report = auditor.audit()
    print(f"Total DoF: {report['total_dof']}, Adjusted Sharpe: {report['adj_sharpe']:.2f}")
"""

from dataclasses import dataclass, field


@dataclass
class DOFCount:
    category: str
    count: int
    items: list = field(default_factory=list)


class DOFAudit:
    """Audit degrees of freedom in a trading system configuration."""

    STRATEGY_DOF = {
        "mean_reversion": {
            "period": "Window length",
            "std_threshold": "Entry threshold",
            "use_adaptive": "Adaptive mode switch",
            "use_trailing_stop": "Trailing stop toggle",
            "trail_atr_mult": "Trailing stop multiplier",
            "use_vol_calibrated_stop": "Vol stop toggle",
            "vol_stop_base_mult": "Vol stop base",
            "vol_stop_min_mult": "Vol stop min",
            "vol_stop_max_mult": "Vol stop max",
            "long_only": "Direction toggle",
        },
        "momentum_breakout": {
            "breakout_period": "Lookback window",
            "volume_multiplier": "Volume threshold",
            "trail_atr_mult": "Trailing stop multiplier",
            "confirm_bars": "Confirmation bars",
            "min_strength": "Minimum strength",
            "use_adaptive_volume": "Adaptive volume toggle",
            "use_vol_calibrated_stop": "Vol stop toggle",
            "vol_stop_base_mult": "Vol stop base",
            "vol_stop_min_mult": "Vol stop min",
            "vol_stop_max_mult": "Vol stop max",
        },
        "trend_following": {
            "fast_ema": "Fast EMA period",
            "slow_ema": "Slow EMA period",
            "trail_atr_mult": "Trailing stop multiplier",
            "use_vol_calibrated_stop": "Vol stop toggle",
            "vol_stop_base_mult": "Vol stop base",
            "vol_stop_min_mult": "Vol stop min",
            "vol_stop_max_mult": "Vol stop max",
        },
    }

    FILTER_DOF = {
        "trend_filter": {"period": "Trend filter lookback"},
        "volatility_regime": {"quantile_high": "Vol high quantile", "quantile_low": "Vol low quantile"},
        "time_filter": {"start_hour": "Start hour", "end_hour": "End hour"},
        "volume_filter": {"multiplier": "Volume multiplier"},
        "adx_filter": {"threshold": "ADX threshold", "period": "ADX period"},
        "sentiment_filter": {"min_score": "Min sentiment score"},
        "rsi_filter": {"period": "RSI period", "oversold": "Oversold level", "overbought": "Overbought level"},
    }

    RISK_DOF = {
        "max_risk_per_trade_pct": "Risk per trade %",
        "max_exposure_pct": "Max exposure %",
        "atr_period": "ATR period",
        "max_concurrent_positions": "Max positions",
        "correlation_reduction": "Correlation reduction",
    }

    EXIT_DOF = {
        "trailing_stop": "Trailing stop toggle",
        "take_profit": "Take profit toggle",
        "time_stop": "Time stop toggle",
    }

    def __init__(
        self,
        strategy: str,
        strategy_params: dict,
        entry_filters: list[str] = None,
        exit_rules: list[str] = None,
        risk_config: dict = None,
    ):
        self.strategy = strategy
        self.strategy_params = strategy_params
        self.entry_filters = entry_filters or []
        self.exit_rules = exit_rules or []
        self.risk_config = risk_config or {}

    def audit(self) -> dict:
        """Run full degrees-of-freedom audit."""
        categories = []

        # 1. Strategy parameters
        strat_dof = 0
        strat_items = []
        param_defs = self.STRATEGY_DOF.get(self.strategy, {})
        for key in param_defs:
            if key in self.strategy_params and self.strategy_params[key] is not None:
                strat_dof += 1
                strat_items.append(f"{param_defs[key]} ({key})")
        categories.append(DOFCount("strategy_params", strat_dof, strat_items))

        # 2. Entry filters
        filter_dof = 0
        filter_items = []
        for flt in self.entry_filters:
            if flt in self.FILTER_DOF:
                # Each filter adds 1 DoF for the filter itself, plus params
                filter_dof += 1
                filter_items.append(f"Entry filter: {flt}")
        categories.append(DOFCount("entry_filters", filter_dof, filter_items))

        # 3. Risk parameters
        risk_dof = 0
        risk_items = []
        for key, desc in self.RISK_DOF.items():
            if key in self.risk_config:
                risk_dof += 1
                risk_items.append(f"{desc} ({key})")
        categories.append(DOFCount("risk_params", risk_dof, risk_items))

        # 4. Exit rules
        exit_dof = 0
        exit_items = []
        for rule in self.exit_rules:
            if rule in self.EXIT_DOF:
                exit_dof += 1
                exit_items.append(rule)
        categories.append(DOFCount("exit_rules", exit_dof, exit_items))

        # 5. Regime filter (if enabled)
        regime_dof = 0
        if self.strategy_params.get("use_regime_filter"):
            regime_dof = 2  # n_states + lookback + prob_threshold
            categories.append(DOFCount("regime_filter", 2, ["HMM n_states", "lookback"]))
        else:
            categories.append(DOFCount("regime_filter", 0, ["disabled"]))

        total_dof = sum(c.count for c in categories)

        # Estimate search-width bias (aligrithm 3.16)
        # Each discrete parameter value tested multiplies the search space
        search_width = 1
        param_values_tested = 0
        for key in param_defs:
            if key in self.strategy_params:
                param_values_tested += 1
                # Estimate typical grid size
                if isinstance(self.strategy_params[key], (int, float)):
                    search_width *= 5  # ~5 values tested per parameter

        return {
            "strategy": self.strategy,
            "total_dof": total_dof,
            "search_width_estimate": search_width,
            "param_values_tested": param_values_tested,
            "categories": {c.category: {"count": c.count, "items": c.items} for c in categories},
            "verdict": self._verdict(total_dof, search_width),
        }

    def _verdict(self, total_dof: int, search_width: int) -> str:
        if total_dof <= 5:
            return "LOW — reasonable parameterization"
        elif total_dof <= 10:
            return "MODERATE — may overfit with < 500 bars of data"
        elif total_dof <= 15:
            return "HIGH — overfitting likely with < 2000 bars"
        else:
            return "CRITICAL — overfitting almost certain"

    def print_report(self, result: dict) -> None:
        """Print formatted audit report."""
        print("=" * 60)
        print(f"DEGREES OF FREEDOM AUDIT — {result['strategy']}")
        print("=" * 60)
        for cat, info in result["categories"].items():
            if info["items"]:
                print(f"  {cat}: {info['count']} DoF")
                for item in info["items"]:
                    print(f"    - {item}")
        print()
        print(f"  Total DoF:          {result['total_dof']}")
        print(f"  Search width:        ~{result['search_width_estimate']:,} combinations")
        print(f"  Verdict:             {result['verdict']}")
        print("=" * 60)
