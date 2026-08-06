"""
Encodes prop firm rules as hard pass/fail gates.
Takes a backtest's trade log + equity curve, returns a verdict.
"""

PROP_FIRM_RULES = {
    "ftmo_2step_phase1": {
        "profit_target_pct": 10.0,
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 10.0,
        "drawdown_type": "static",       # anchored to initial balance, never moves
        "min_trading_days": 4,
        "max_single_trade_profit_share_pct": 30.0,  # consistency guard, conservative estimate
    },
    "ftmo_2step_phase2": {
        "profit_target_pct": 5.0,
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 10.0,
        "drawdown_type": "static",
        "min_trading_days": 4,
        "max_single_trade_profit_share_pct": 30.0,
    },
    "the5ers_high_stakes": {
        "profit_target_pct": 8.0,        # mid-point of their 6-10% range; adjust to your exact plan
        "max_daily_loss_pct": 4.0,
        "max_total_drawdown_pct": 6.0,
        "drawdown_type": "static",       # anchored to starting balance
        "min_trading_days": 3,
        "max_single_day_profit_share_pct": 50.0,  # their actual stated consistency rule
    },
   "fundingpips_2step_phase1": {
        "profit_target_pct": 8.0,        # or 10.0 if you pick that option — adjust to your exact purchase
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 10.0,
        "drawdown_type": "static",
        "min_trading_days": 3,
        "max_single_trade_profit_share_pct": None,  # no consistency rule on standard 2-Step
    },
    "fundingpips_2step_phase2": {
        "profit_target_pct": 5.0,
        "max_daily_loss_pct": 5.0,
        "max_total_drawdown_pct": 10.0,
        "drawdown_type": "static",
        "min_trading_days": 3,
        "max_single_trade_profit_share_pct": None,
    },
}


def evaluate_against_rules(stats, daily_pnl_series, firm_key):
    """
    stats: the backtesting.py stats object/dict (Return %, Max. Drawdown %, # Trades, etc.)
    daily_pnl_series: pandas Series of daily P&L, indexed by date
    firm_key: one of PROP_FIRM_RULES keys
    Returns dict: {"passed": bool, "failures": [...], "warnings": [...]}
    """
    rules = PROP_FIRM_RULES[firm_key]
    failures = []
    warnings = []

    # 1. Profit target
    total_return = stats["Return [%]"]
    if total_return < rules["profit_target_pct"]:
        failures.append(
            f"Return {total_return:.2f}% below profit target {rules['profit_target_pct']}%"
        )

    # 2. Max total drawdown (static, from initial balance)
    max_dd = abs(stats["Max. Drawdown [%]"])
    if max_dd > rules["max_total_drawdown_pct"]:
        failures.append(
            f"Max drawdown {max_dd:.2f}% exceeds limit {rules['max_total_drawdown_pct']}%"
        )
    elif max_dd > rules["max_total_drawdown_pct"] * 0.8:
        warnings.append(
            f"Max drawdown {max_dd:.2f}% is close (>80%) to the {rules['max_total_drawdown_pct']}% limit"
        )

    # 3. Daily loss limit — needs per-day P&L, not just overall stats
    worst_day_pct = daily_pnl_series.min()
    if worst_day_pct < -rules["max_daily_loss_pct"]:
        failures.append(
            f"Worst single day {worst_day_pct:.2f}% breaches daily loss limit "
            f"-{rules['max_daily_loss_pct']}%"
        )

    # 4. Minimum trading days
    active_days = (daily_pnl_series != 0).sum()
    if active_days < rules["min_trading_days"]:
        failures.append(
            f"Only {active_days} active trading days, needs {rules['min_trading_days']}"
        )

    return {
        "firm": firm_key,
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "worst_day_pct": worst_day_pct,
    }
