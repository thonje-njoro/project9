"""
monte_carlo_sizing.py — Bootstrap optimal position sizing for prop firm constraints.

Monte Carlo simulation to find the optimal risk per trade and number of concurrent
positions that maximizes expected return while keeping daily DD breach < 5%.

Usage:
    from risk.monte_carlo_sizing import optimize_risk_concurrent
    results, optimal = optimize_risk_concurrent(trade_returns)
    print(optimal["recommendation"])
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Optional


def simulate_prop_firm_run(
    trade_returns: np.ndarray,
    risk_per_trade: float = 0.02,
    max_concurrent: int = 2,
    daily_dd_limit: float = 0.04,
    max_dd_limit: float = 0.10,
    initial_capital: float = 50_000,
    max_trades: int = 200,
    profit_target_pct: float = 0.10,
) -> dict:
    """
    Simulate one prop firm attempt with given risk parameters.

    Each simulation day: 1 to max_concurrent trades are drawn with replacement
    from the historical trade return distribution. Daily DD and trailing max DD
    are tracked. The run ends when:
      - Capital hits 10% profit target → WIN
      - Daily drawdown > 4% → LOSS
      - Trailing drawdown > 10% → LOSS
      - Max trades exhausted → TIMEOUT
    """
    capital = initial_capital
    peak = capital
    trades_executed = 0
    daily_dd_breach = False
    hit_target = False
    survived = True
    max_dd = 0.0

    for day in range(200):  # max 200 days (shouldn't reach this)
        if trades_executed >= max_trades:
            break

        today_trades = np.random.randint(1, max_concurrent + 1)
        day_pnl = 0.0

        for _ in range(today_trades):
            if trades_executed >= max_trades:
                break
            r = np.random.choice(trade_returns)
            trade_pnl = capital * risk_per_trade * r
            day_pnl += trade_pnl
            trades_executed += 1

        # Check daily DD
        if day_pnl < -capital * daily_dd_limit:
            daily_dd_breach = True
            survived = False
            break

        capital += day_pnl
        if capital > peak:
            peak = capital

        # Check max DD
        current_dd = (peak - capital) / peak
        max_dd = max(max_dd, current_dd)
        if current_dd > max_dd_limit:
            survived = False
            break

        # Check profit target
        if (capital - initial_capital) / initial_capital >= profit_target_pct:
            hit_target = True
            break

    return {
        'survived': survived,
        'hit_target': hit_target,
        'max_dd': max_dd,
        'daily_dd_breach': daily_dd_breach,
        'final_capital': capital,
        'peak_capital': peak,
        'trades_executed': trades_executed,
    }


def optimize_risk_concurrent(
    trade_returns: np.ndarray,
    risk_values: Optional[list] = None,
    concurrent_values: Optional[list] = None,
    n_simulations: int = 5000,
    daily_dd_limit: float = 0.04,
    max_dd_limit: float = 0.10,
    initial_capital: float = 50_000,
    profit_target_pct: float = 0.10,
    max_daily_dd_breach_prob: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Grid search over (risk_per_trade, max_concurrent) pairs.

    Args:
        trade_returns: Array of historical per-trade returns (fractional)
        risk_values: Risk per trade values to test (default: 0.5%-4%)
        concurrent_values: Max concurrent positions to test (default: 1, 2, 3)
        n_simulations: Monte Carlo paths per configuration

    Returns:
        (results_df, optimal_config_dict)
    """
    if risk_values is None:
        risk_values = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175,
                       0.02, 0.025, 0.03, 0.035, 0.04]
    if concurrent_values is None:
        concurrent_values = [1, 2, 3]

    rows = []

    for risk, conc in product(risk_values, concurrent_values):
        survived = hit_target = dd_breach = 0
        final_caps = []

        for _ in range(n_simulations):
            sim = simulate_prop_firm_run(
                trade_returns=trade_returns,
                risk_per_trade=risk,
                max_concurrent=conc,
                daily_dd_limit=daily_dd_limit,
                max_dd_limit=max_dd_limit,
                initial_capital=initial_capital,
                profit_target_pct=profit_target_pct,
            )
            if sim['survived']:
                survived += 1
            if sim['hit_target']:
                hit_target += 1
            if sim['daily_dd_breach']:
                dd_breach += 1
            final_caps.append(sim['final_capital'])

        survival_rate = survived / n_simulations
        target_hit_rate = hit_target / n_simulations
        dd_breach_prob = dd_breach / n_simulations

        # Risk score: target hit rate - 2x DD breach (penalizes DD risk)
        risk_score = target_hit_rate - 2 * dd_breach_prob

        rows.append({
            'risk_pct': risk * 100,
            'concurrent': conc,
            'survival_rate': survival_rate,
            'target_hit_rate': target_hit_rate,
            'dd_breach_prob': dd_breach_prob,
            'avg_final_equity': np.mean(final_caps),
            'risk_score': risk_score,
        })

    df = pd.DataFrame(rows)

    # Find optimal: max risk_score where DD breach prob < threshold
    safe = df[
        (df['dd_breach_prob'] <= max_daily_dd_breach_prob)
    ]
    if not safe.empty:
        best = safe.loc[safe['risk_score'].idxmax()]
        recommendation = (
            f"OPTIMAL: risk={best['risk_pct']:.1f}%, concurrent={best['concurrent']:.0f} — "
            f"target={best['target_hit_rate']:.1%}, DD breach={best['dd_breach_prob']:.1%}"
        )
    else:
        # Fallback: lowest DD breach probability
        best = df.loc[df['dd_breach_prob'].idxmin()]
        recommendation = (
            f"FALLBACK (no config safe): risk={best['risk_pct']:.1f}%, concurrent={best['concurrent']:.0f} — "
            f"DD breach={best['dd_breach_prob']:.1%} (exceeds 5%)"
        )

    optimal = {
        'risk_per_trade_pct': best['risk_pct'] / 100,
        'max_concurrent': int(best['concurrent']),
        'target_hit_rate': best['target_hit_rate'],
        'dd_breach_prob': best['dd_breach_prob'],
        'recommendation': recommendation,
    }

    # Print results table
    print(f"\n{'='*72}")
    print("MONTE CARLO SIZING OPTIMIZATION")
    print(f"{'='*72}")
    print(f"  Trade returns: n={len(trade_returns)}")
    print(f"  Simulations per config: {n_simulations}")
    print(f"  Daily DD limit: {daily_dd_limit:.0%}")
    print(f"  Max DD limit: {max_dd_limit:.0%}")
    print(f"  Profit target: {profit_target_pct:.0%}")
    print(f"{'='*72}")
    print(f"  {'Risk%':>6s} {'Conc':>5s} {'Survival':>9s} {'Target':>9s} {'DD Breach':>9s} {'RiskScore':>9s}")
    print(f"  {'-'*6} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for _, r in df.iterrows():
        print(f"  {r['risk_pct']:5.1f}% {r['concurrent']:5d} "
              f"{r['survival_rate']:8.1%} "
              f"{r['target_hit_rate']:8.1%} "
              f"{r['dd_breach_prob']:8.1%} "
              f"{r['risk_score']:8.4f}")

    print(f"\n  {recommendation}")

    return df, optimal
