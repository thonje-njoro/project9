import sys
sys.path.insert(0, ".")

import pandas as pd
from backtesting import Backtest
from strategies.baseline_sma_cross import SmaCross, df  # reuse what's already there
from evaluator.daily_pnl import compute_daily_pnl_pct
from evaluator.prop_firm_rules import evaluate_against_rules

INITIAL_CASH = 10_000

bt = Backtest(df, SmaCross, cash=INITIAL_CASH, commission=0.0002, margin=1/30, finalize_trades=True)
stats = bt.run()

daily_pnl = compute_daily_pnl_pct(stats["_equity_curve"], INITIAL_CASH)

for firm_key in ["ftmo_2step_phase1", "the5ers_high_stakes", "fundingpips_2step_phase1"]:
    result = evaluate_against_rules(stats, daily_pnl, firm_key)
    print(f"\n=== {firm_key} ===")
    print(f"Passed: {result['passed']}")
    print(f"Return: {result['return_pct']:.2f}%  |  Max DD: {result['max_drawdown_pct']:.2f}%  |  Worst day: {result['worst_day_pct']:.2f}%")
    if result["failures"]:
        print("Failures:")
        for f in result["failures"]:
            print(f"  - {f}")
    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")
