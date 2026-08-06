"""Prop firm rule simulator with enhanced features.

Features:
1. FTMO-style daily/max drawdown checks
2. Daily profit cap at 30% of target (consistency rule)
3. Minimum trading day counter (FTMO: 10 days)
4. Time-to-target estimator
5. Progressive risk reduction triggers
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PropFirmResult:
    passed: bool
    failure_reason: Optional[str] = None
    failure_date: Optional[str] = None
    daily_drawdown_breaches: int = 0
    consistency_triggers: int = 0
    daily_profit_caps_hit: int = 0
    trading_days_count: int = 0
    min_trading_days_met: bool = False
    profit_target_pct: float = 0.0
    profit_achieved_pct: float = 0.0
    estimated_days_to_target: Optional[int] = None
    progressive_risk_triggered: bool = False
    max_consecutive_losses: int = 0
    adjusted_equity_curve: pd.Series = field(default_factory=pd.Series)


def simulate_prop_firm_rules(
    portfolio,
    rules: dict,
    profit_target_pct: float = 0.10,
    daily_profit_cap_pct: float = 0.30,
    min_trading_days: int = 10,
    progressive_risk_thresholds: list = None,
) -> PropFirmResult:
    """
    Enhanced prop firm rule simulation.

    Args:
        portfolio: vbt.Portfolio object
        rules: dict with 'daily_drawdown_pct', 'max_drawdown_pct',
               'consistency_warn_pct', 'consistency_block_pct'
        profit_target_pct: Total profit target (e.g. 0.10 for 10%)
        daily_profit_cap_pct: Max portion of target per day (e.g. 0.30 = 30%)
        min_trading_days: Minimum distinct trading days required
        progressive_risk_thresholds: [(losses, reduction), ...] e.g.
            [(2, 0.75), (4, 0.5), (6, 0.25)]
    """
    if progressive_risk_thresholds is None:
        progressive_risk_thresholds = [(2, 0.75), (4, 0.5), (6, 0.25)]

    equity = portfolio.value()
    daily_equity = equity.resample("1D").last().dropna()

    if len(daily_equity) == 0:
        return PropFirmResult(
            passed=False,
            failure_reason="no_data",
            trading_days_count=0,
        )

    initial = daily_equity.iloc[0]
    daily_start = daily_equity.shift(1).fillna(initial)
    running_max = daily_equity.cummax()

    # === 1. Total drawdown check ===
    total_dd = (daily_equity - running_max) / running_max
    dd_breach_idx = total_dd[total_dd < -rules["max_drawdown_pct"]].first_valid_index()
    if dd_breach_idx is not None:
        return _build_failure("max_drawdown", str(dd_breach_idx), daily_equity)

    # === 2. Daily drawdown check ===
    daily_dd = (daily_equity - daily_start) / daily_start
    daily_breaches = int((daily_dd < -rules["daily_drawdown_pct"]).sum())
    dd_daily_breach = daily_dd[daily_dd < -rules["daily_drawdown_pct"]].first_valid_index()
    if dd_daily_breach is not None:
        return _build_failure("daily_drawdown", str(dd_daily_breach), daily_equity,
                              daily_breaches=daily_breaches)

    # === 3. Minimum trading days check ===
    trading_days = len(daily_equity)
    min_days_met = trading_days >= min_trading_days

    # === 4. Profit target tracking ===
    total_profit = (daily_equity.iloc[-1] - initial)
    profit_achieved = total_profit / initial
    profit_target_actual = profit_target_pct * initial

    # === 5. Daily profit cap (consistency rule) ===
    daily_profit = daily_equity.diff().fillna(0)
    max_daily_allowed = profit_target_actual * daily_profit_cap_pct
    daily_caps_hit = int((daily_profit.abs() > max_daily_allowed).sum())

    # === 6. Consistency ratio (no single day dominates) ===
    cumulative_profit = total_profit
    triggers = 0
    if cumulative_profit != 0:
        consistency_ratio = daily_profit / cumulative_profit
        triggers = int((consistency_ratio > rules["consistency_block_pct"]).sum())

    # === 7. Consecutive losses tracker ===
    daily_returns = daily_equity.pct_change().fillna(0)
    consec_losses = 0
    max_consec = 0
    for ret in daily_returns:
        if ret < 0:
            consec_losses += 1
            max_consec = max(max_consec, consec_losses)
        else:
            consec_losses = 0

    # === 8. Progressive risk reduction ===
    risk_reduced = False
    for loss_threshold, reduction in progressive_risk_thresholds:
        if max_consec >= loss_threshold:
            risk_reduced = True
            break

    # === 9. Time-to-target estimator ===
    est_days = None
    if trading_days > 5 and profit_achieved > 0:
        daily_avg_profit = total_profit / trading_days
        if daily_avg_profit > 0:
            remaining = profit_target_actual - total_profit
            est_days = max(1, int(remaining / daily_avg_profit))
    elif trading_days > 5 and profit_achieved <= 0:
        est_days = 999  # Not on track

    return PropFirmResult(
        passed=True,
        daily_drawdown_breaches=daily_breaches,
        consistency_triggers=triggers,
        daily_profit_caps_hit=daily_caps_hit,
        trading_days_count=trading_days,
        min_trading_days_met=min_days_met,
        profit_target_pct=profit_target_pct,
        profit_achieved_pct=profit_achieved * 100,
        estimated_days_to_target=est_days,
        progressive_risk_triggered=risk_reduced,
        max_consecutive_losses=max_consec,
        adjusted_equity_curve=daily_equity,
    )


def _build_failure(reason: str, date: str, equity: pd.Series,
                   daily_breaches: int = 0) -> PropFirmResult:
    return PropFirmResult(
        passed=False,
        failure_reason=reason,
        failure_date=date,
        daily_drawdown_breaches=daily_breaches,
        trading_days_count=int(len(equity)),
        adjusted_equity_curve=equity,
    )


def print_prop_firm_report(result: PropFirmResult, label: str = "") -> None:
    """Print a formatted prop firm simulation report."""
    prefix = f" [{label}]" if label else ""
    print(f"--- Prop Firm Simulation{prefix} ---")
    print(f"  Status:              {'PASS' if result.passed else 'FAIL'}")
    if result.failure_reason:
        print(f"  Failure:             {result.failure_reason} on {result.failure_date}")

    print(f"  Trading days:        {result.trading_days_count}")
    print(f"  Min days met:        {'YES' if result.min_trading_days_met else 'NO'}")
    print(f"  Profit achieved:     {result.profit_achieved_pct:.2f}% (target: {result.profit_target_pct*100:.0f}%)")

    if result.estimated_days_to_target is not None:
        if result.estimated_days_to_target < 999:
            print(f"  Est. days to target: {result.estimated_days_to_target}")
        else:
            print(f"  Est. days to target: NOT ON TRACK (negative avg return)")

    print(f"  Daily DD breaches:   {result.daily_drawdown_breaches}")
    print(f"  Consistency triggers:{result.consistency_triggers}")
    print(f"  Profit caps hit:     {result.daily_profit_caps_hit}")
    print(f"  Max consec losses:   {result.max_consecutive_losses}")
    print(f"  Risk reduction:      {'ACTIVE' if result.progressive_risk_triggered else 'NONE'}")
    print()


# ════════════════════════════════════════════════════════════════
# Prop Firm Challenge Sprint Simulator (separate from slow system)
# ════════════════════════════════════════════════════════════════

@dataclass
class SprintResult:
    """Result of a challenge sprint simulation."""
    passed: bool
    failure_reason: Optional[str] = None
    failure_day: Optional[int] = None
    profit_achieved_pct: float = 0.0
    profit_target_pct: float = 0.10
    days_elapsed: int = 0
    daily_dd_breaches: int = 0
    max_drawdown_pct: float = 0.0
    trades_executed: int = 0
    consistency_triggers: int = 0
    final_equity: float = 50_000
    peak_equity: float = 50_000
    daily_equity_curve: pd.Series = field(default_factory=pd.Series)
    phases: list = field(default_factory=list)


def simulate_challenge_sprint(
    portfolio,
    initial_capital: float = 50_000,
    profit_target_pct: float = 0.10,
    daily_dd_limit_pct: float = 0.035,
    max_dd_limit_pct: float = 0.09,
    max_trading_days: int = 22,
    consistency_max_day_pct: float = 0.30,
) -> SprintResult:
    """Simulate a prop firm challenge sprint run with phase tracking.

    This is a more focused, sprint-specific simulator that tracks:
      - Daily equity (resampled from portfolio value)
      - Drawdown from intraday peak (not just daily close)
      - Daily P&L against the 4% daily loss limit
      - Phase progression (probing to acceleration to preservation)
      - Consistency rule (single-day profit cap)

    Args:
        portfolio: vbt.Portfolio object from the sprint backtest
        initial_capital: Starting account balance
        profit_target_pct: 10% profit target
        daily_dd_limit_pct: 3.5% daily loss limit (buffer below FTMO's 4%)
        max_dd_limit_pct: 9% max drawdown (buffer below FTMO's 10%)
        max_trading_days: 22 trading day limit
        consistency_max_day_pct: Max 30% of total profit in one day

    Returns:
        SprintResult with pass/fail status and full diagnostics
    """
    equity = portfolio.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]

    daily_equity = equity.resample("1D").last().dropna()
    if len(daily_equity) == 0:
        return SprintResult(passed=False, failure_reason="no_data")

    initial = float(daily_equity.iloc[0])
    peak = initial
    capital = initial
    daily_dd_breaches = 0
    daily_profits = []
    max_dd = 0.0

    # Track peak drawdown from all intraday equity values
    all_equity = equity.values
    all_peak = np.maximum.accumulate(all_equity)
    all_dd = (all_equity - all_peak) / all_peak
    max_dd = float(abs(all_dd.min())) if len(all_dd) > 0 else 0

    for day_idx in range(min(len(daily_equity), max_trading_days + 1)):
        day_equity = float(daily_equity.iloc[day_idx]) if day_idx < len(daily_equity) else capital

        if day_idx == 0:
            continue

        day_pnl = day_equity - capital
        day_return_pct = day_pnl / capital if capital > 0 else 0
        daily_profits.append(day_pnl)

        # Check daily drawdown
        if day_return_pct < -daily_dd_limit_pct:
            daily_dd_breaches += 1
            return SprintResult(
                passed=False,
                failure_reason="daily_drawdown_breach",
                failure_day=day_idx,
                profit_achieved_pct=(day_equity - initial) / initial * 100,
                profit_target_pct=profit_target_pct,
                days_elapsed=day_idx,
                daily_dd_breaches=daily_dd_breaches,
                max_drawdown_pct=max_dd * 100,
                trades_executed=int(portfolio.trades.count()),
                final_equity=day_equity,
                peak_equity=peak,
                daily_equity_curve=daily_equity.iloc[:day_idx + 1],
            )

        # Track peak
        if day_equity > peak:
            peak = day_equity

        # Check max drawdown
        current_dd = (peak - day_equity) / peak if peak > 0 else 0
        if current_dd > max_dd_limit_pct:
            return SprintResult(
                passed=False,
                failure_reason="max_drawdown_breach",
                failure_day=day_idx,
                profit_achieved_pct=(day_equity - initial) / initial * 100,
                profit_target_pct=profit_target_pct,
                days_elapsed=day_idx,
                daily_dd_breaches=daily_dd_breaches,
                max_drawdown_pct=current_dd * 100,
                trades_executed=int(portfolio.trades.count()),
                final_equity=day_equity,
                peak_equity=peak,
                daily_equity_curve=daily_equity.iloc[:day_idx + 1],
            )

        capital = day_equity

        # Check profit target
        total_profit_pct = (capital - initial) / initial
        if total_profit_pct >= profit_target_pct:
            return SprintResult(
                passed=True,
                profit_achieved_pct=total_profit_pct * 100,
                profit_target_pct=profit_target_pct,
                days_elapsed=day_idx,
                daily_dd_breaches=daily_dd_breaches,
                max_drawdown_pct=max_dd * 100,
                trades_executed=int(portfolio.trades.count()),
                final_equity=capital,
                peak_equity=peak,
                daily_equity_curve=daily_equity.iloc[:day_idx + 1],
            )

        if day_idx >= max_trading_days:
            break

    total_profit_pct = (capital - initial) / initial

    # Consistency rule check
    consistency_pass = True
    if len(daily_profits) >= 3:
        total_profit = sum(daily_profits)
        if total_profit > 0:
            max_day = max(daily_profits)
            max_ratio = max_day / total_profit
            if max_ratio > consistency_max_day_pct:
                consistency_pass = False

    passed = (
        total_profit_pct >= profit_target_pct
        and consistency_pass
        and max_dd < max_dd_limit_pct
        and daily_dd_breaches == 0
    )

    if not passed and day_idx >= max_trading_days and total_profit_pct < profit_target_pct:
        reason = "time_limit_not_profitable"
    elif not passed and not consistency_pass:
        reason = "consistency_rule"
    elif not passed:
        reason = "unknown"
    else:
        reason = None

    return SprintResult(
        passed=passed,
        failure_reason=reason,
        failure_day=day_idx,
        profit_achieved_pct=total_profit_pct * 100,
        profit_target_pct=profit_target_pct,
        days_elapsed=day_idx,
        daily_dd_breaches=daily_dd_breaches,
        max_drawdown_pct=max_dd * 100,
        trades_executed=int(portfolio.trades.count()),
        final_equity=capital,
        peak_equity=peak,
        daily_equity_curve=daily_equity,
    )


def print_sprint_report(result: SprintResult, label: str = "") -> None:
    """Print a formatted sprint simulation report."""
    prefix = f" [{label}]" if label else ""
    print(f"=== Challenge Sprint Simulation{prefix} ===")
    print(f"  Status:              {'PASS' if result.passed else 'FAIL'}")
    if result.failure_reason:
        print(f"  Failure:             {result.failure_reason}"
              f"{' on day ' + str(result.failure_day) if result.failure_day else ''}")
    print(f"  Trading days:        {result.days_elapsed}")
    print(f"  Profit achieved:     {result.profit_achieved_pct:.2f}% "
          f"(target: {result.profit_target_pct*100:.0f}%)")
    print(f"  Max drawdown:        {result.max_drawdown_pct:.2f}%")
    print(f"  Daily DD breaches:   {result.daily_dd_breaches}")
    print(f"  Trades executed:     {result.trades_executed}")
    print(f"  Final equity:        ${result.final_equity:,.2f}")
    print(f"  Peak equity:         ${result.peak_equity:,.2f}")
    print()
