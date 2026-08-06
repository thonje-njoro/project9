"""
grid_search_cper_gld.py — Systematic parameter optimization for CPER_GLD_RATIO.

Tests 4D grid: z_entry × z_exit × window × trail_atr_mult × z_take_profit
Evaluates on: PF, trade count, Sharpe, max DD
Target: PF > 1.5 with > 80 trades

Usage:
    from optimization.grid_search_cper_gld import grid_search_cper_gld
    results = grid_search_cper_gld(ratio_df, gld_df, BACKTEST_CONFIG)
"""

import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path


def grid_search_cper_gld(
    ratio_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Full grid search over CPER_GLD_RATIO parameters.

    Grid (4D + 1):
      z_entry:         [1.5, 2.0, 2.5, 3.0, 3.5]
      z_exit:          [0.0, 0.3, 0.5]
      window:          [20, 30, 40, 60]
      trail_atr_mult:  [1.0, 1.5, 2.0, 3.0]
      z_take_profit:   [0.0, 0.5]

    Total: 5 × 3 × 4 × 4 × 2 = 480 configs (minus invalid combos where z_exit >= z_entry)
    """
    from engine import BacktestEngine
    from reporting.metrics import _compute_sharpe

    z_entry_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    z_exit_values = [0.0, 0.3, 0.5]
    window_values = [20, 30, 40, 60]
    trail_values = [1.0, 1.5, 2.0, 3.0]
    tp_values = [0.0, 0.5]

    results = []
    total_configs = 0

    for z_entry, z_exit, window, trail, tp in product(
        z_entry_values, z_exit_values, window_values, trail_values, tp_values
    ):
        if z_exit >= z_entry:
            continue

        total_configs += 1
        params = {
            "z_entry": z_entry,
            "z_exit": z_exit,
            "z_take_profit": tp,
            "window": window,
            "use_trailing_stop": True,
            "trail_atr_mult": trail,
        }

        try:
            engine = BacktestEngine(
                {"CPER_GLD_RATIO": ratio_df},
                config,
                use_regime_filter=False,
            )
            engine.kelly_factors = {"CPER_GLD_RATIO": 1.0}
            # Monkey-patch params for this run
            from config import STRATEGY_PARAMS
            STRATEGY_PARAMS["cper_gld_ratio"]["CPER_GLD_RATIO"] = params

            pf = engine._run_single("CPER_GLD_RATIO", ratio_df, kelly_mult=1.0)

            trades = pf.trades.count()
            if trades < 10:
                continue

            sharpe = _compute_sharpe(pf, config["risk_free_rate"])
            profit_factor = pf.trades.profit_factor() if trades > 0 else 0.0
            win_rate = pf.trades.win_rate() if trades > 0 else 0.0
            total_ret = pf.total_return() * 100
            max_dd = pf.max_drawdown() * 100

            results.append({
                "z_entry": z_entry,
                "z_exit": z_exit,
                "z_tp": tp,
                "window": window,
                "trail": trail,
                "trades": trades,
                "sharpe": f"{sharpe:.3f}",
                "pf": f"{profit_factor:.3f}",
                "win_rate": f"{win_rate:.1%}",
                "return_pct": f"{total_ret:.2f}",
                "max_dd": f"{max_dd:.2f}",
                # Composite score: PF * sqrt(trades)
                "score": profit_factor * np.sqrt(trades) if profit_factor > 0 else 0,
            })
        except Exception as e:
            pass

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("ERROR: No valid results from grid search")
        return results_df

    # Convert numeric columns back
    for c in ["sharpe", "pf", "win_rate", "return_pct", "max_dd"]:
        results_df[c] = pd.to_numeric(results_df[c], errors="coerce")

    # Print summary
    print(f"\n{'='*72}")
    print(f"CPER_GLD_RATIO GRID SEARCH RESULTS")
    print(f"Configs tested: {total_configs}")
    print(f"Valid results: {len(results_df)}")
    print(f"{'='*72}")

    # Top 15 by composite score
    top = results_df.nlargest(15, "score")
    print("\nTop 15 by PF * sqrt(trades):")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Best PF with > 80 trades
    good = results_df[
        (results_df["pf"] > 1.5) & (results_df["trades"] > 80)
    ].nlargest(10, "pf")
    if not good.empty:
        print("\nBest PF > 1.5 with > 80 trades:")
        print(good.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\nNo config achieved PF > 1.5 with > 80 trades.")
        best_tradeoff = results_df.nlargest(10, "score")
        print("Best trade-offs (PF * sqrt(trades)):")
        print(best_tradeoff.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "grid_search_cper_gld.csv", index=False)
    print(f"\nResults saved to results/grid_search_cper_gld.csv")

    return results_df
