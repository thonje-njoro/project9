"""
prop_firm/rule_simulator.py — FTMO/The5ers/FundingPips rule simulation.
Walks equity curve day by day to check if prop firm rules would be violated.
"""

import logging

import numpy as np
import pandas as pd

from backtest.config import PROP_FIRM_RULES

logger = logging.getLogger(__name__)


def simulate_prop_firm(equity_curve: pd.Series,
                       rules: dict,
                       initial_equity: float = 50_000) -> dict:
    """
    Simulate prop firm rules against an equity curve.

    Args:
        equity_curve: Daily equity series.
        rules: Prop firm rule dict with keys:
            profit_target_pct, max_drawdown_pct, daily_loss_pct,
            min_trading_days, daily_profit_cap_pct.
        initial_equity: Starting equity.

    Returns:
        dict with:
            passes: bool
            failure_reason: str or None
            failure_date: date or None
            days_to_target: int or None
            consistency_violations: int
            max_drawdown_pct: float
            max_daily_loss_pct: float
    """
    if equity_curve.empty:
        return {"passes": False, "failure_reason": "empty_equity_curve"}

    profit_target = rules["profit_target_pct"]
    max_dd = rules["max_drawdown_pct"]
    daily_loss_limit = rules["daily_loss_pct"]
    min_days = rules["min_trading_days"]
    daily_profit_cap = rules.get("daily_profit_cap_pct", 0.30)

    # Ensure daily frequency
    daily = equity_curve.resample("1D").last().dropna()
    if len(daily) < 2:
        return {"passes": False, "failure_reason": "insufficient_data"}

    # Compute daily returns
    daily_returns = daily.pct_change().dropna()

    peak = initial_equity
    consistency_violations = 0
    failure_date = None
    failure_reason = None
    days_to_target = None
    cumulative_profit = 0.0

    for i, (date, ret) in enumerate(daily_returns.items()):
        current_equity = daily.iloc[i + 1]

        # Update peak
        peak = max(peak, current_equity)

        # Check daily loss
        if ret < -daily_loss_limit:
            failure_date = date
            failure_reason = f"daily_loss_{ret:.3%}_exceeds_{daily_loss_limit:.3%}"
            break

        # Check max drawdown from peak
        dd_from_peak = (peak - current_equity) / peak
        if dd_from_peak > max_dd:
            failure_date = date
            failure_reason = f"drawdown_{dd_from_peak:.3%}_exceeds_{max_dd:.3%}"
            break

        # Track cumulative profit for consistency check
        cumulative_profit = (current_equity - initial_equity) / initial_equity

        # Consistency check: no single day should contribute > daily_profit_cap
        # of total profit target
        if ret > daily_profit_cap * profit_target:
            consistency_violations += 1

        # Check if profit target reached
        if cumulative_profit >= profit_target and days_to_target is None:
            days_to_target = i + 1

    # Final checks
    total_return = (daily.iloc[-1] - initial_equity) / initial_equity
    trading_days = len(daily_returns)

    passes = (
        failure_reason is None
        and trading_days >= min_days
        and total_return >= profit_target
    )

    max_dd_pct = float(((daily.cummax() - daily) / daily.cummax()).max())
    max_daily_loss_pct = float(daily_returns.min()) if len(daily_returns) > 0 else 0

    return {
        "passes": passes,
        "failure_reason": failure_reason,
        "failure_date": failure_date,
        "days_to_target": days_to_target,
        "consistency_violations": consistency_violations,
        "max_drawdown_pct": max_dd_pct,
        "max_daily_loss_pct": max_daily_loss_pct,
        "total_return_pct": float(total_return),
        "trading_days": trading_days,
    }


def simulate_all_firms(equity_curve: pd.Series,
                       initial_equity: float = 50_000) -> dict:
    """
    Run simulation against all configured prop firm rules.

    Returns:
        dict: {firm_name: simulation_result}
    """
    results = {}
    for firm_name, rules in PROP_FIRM_RULES.items():
        results[firm_name] = simulate_prop_firm(equity_curve, rules, initial_equity)
        logger.info(
            f"{firm_name}: {'PASS' if results[firm_name]['passes'] else 'FAIL'} "
            f"({results[firm_name].get('failure_reason', 'ok')})"
        )
    return results
