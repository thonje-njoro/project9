"""Performance metrics and reporting.

Includes:
- Standard Sharpe, return, drawdown
- Trade bootstrap confidence intervals (Item 5)
- Kelly fraction reporting (Item 1 output)
- Combined report with warnings
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from config import RISK_CONFIG


def _prop_result(res):
    """Handle both dict and PropFirmResult objects."""
    if hasattr(res, 'passed'):  # PropFirmResult dataclass
        return res.passed, res.failure_reason, res.consistency_triggers, getattr(res, 'profit_achieved_pct', 0)
    return res.get("passed", False), res.get("failure_reason"), res.get("consistency_triggers", 0), res.get("profit_target_pct", 0)


def _compute_sharpe(portfolio, risk_free: float = 0.045) -> float:
    """Compute Sharpe ratio using trade-return-based annualization (CORRECTED).

    The old daily-resample approach produced values like -28.51 for strategies
    with PF=2.90 because most daily returns are zero (strategy trades only ~5%
    of days), making the daily std artificially low and annualization wrong.

    CORRECT METHOD:
      - Extracts per-trade P&L returns from the portfolio
      - Computes average holding period in trading days
      - Annualizes: Sharpe = (mean_return / std_return) * sqrt(252 / avg_hold_days)

    This is the ONLY correct method for low-frequency strategies (4H/1D timeframes).
    For high-frequency strategies (15min/1h), both methods converge.
    """
    trades = portfolio.trades.count()
    if trades < 3:
        return 0.0

    records = portfolio.trades.records_readable
    if records is None or len(records) < 3:
        return 0.0

    # Find the return column (name varies by vectorbt version)
    ret_col = [c for c in records.columns if 'return' in c.lower()]
    if not ret_col:
        return 0.0

    trade_returns = records[ret_col[0]].values.astype(float)
    trade_returns = trade_returns[~np.isnan(trade_returns) & ~np.isinf(trade_returns)]

    if len(trade_returns) < 3:
        return 0.0

    mean_r = trade_returns.mean()
    std_r = trade_returns.std()

    if std_r < 1e-10:
        return 0.0

    # Compute average holding period in trading days
    avg_hold_days = 1.0
    if 'Entry Timestamp' in records.columns and 'Exit Timestamp' in records.columns:
        durations = (records['Exit Timestamp'] - records['Entry Timestamp']).dt.total_seconds()
        # Convert to trading days (86400 sec/day * 5/7 for ~5 trading days per week)
        avg_hold_days = float(durations.mean() / (86400 * 5/7))
        avg_hold_days = max(avg_hold_days, 0.5)  # floor at half a day

    # Annualization factor: sqrt(trading periods per year)
    # Each trade lasts avg_hold_days, so there are ~252/avg_hold_days
    # independent trading opportunities per year
    ann_factor = np.sqrt(252.0 / avg_hold_days)

    # Subtract risk-free per holding period
    rf_per_period = (1 + risk_free) ** (avg_hold_days / 252) - 1
    excess_mean = mean_r - rf_per_period

    sharpe = (excess_mean / std_r) * ann_factor

    return float(sharpe)


def bootstrap_sharpe(
    trade_returns: pd.Series,
    n_sim: int = 10000,
    confidence: float = 0.90,
    avg_holding_days: float = 1.0,
) -> dict:
    """Bootstrap confidence intervals for Sharpe ratio from trade returns.

    Resamples trade returns with replacement to compute the distribution
    of possible Sharpe ratios. This reveals how much sampling noise
    affects your metric — critical when N trades is small.

    Args:
        trade_returns: Series of per-trade P&L returns
        n_sim: Number of bootstrap iterations
        confidence: Confidence level for the interval
        avg_holding_days: Average holding period in trading days.
            Used for annualization: sqrt(252 / avg_holding_days).

    Returns:
        dict with median, lower, upper, and std of bootstrapped Sharpe
    """
    n = len(trade_returns)
    if n < 5:
        return {"median": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0, "n": n}

    # Annualization factor: sqrt(trading periods per year)
    # Trade returns span avg_holding_days each, so there are ~252/avg_holding_days
    # independent trading opportunities per year
    ann_factor = np.sqrt(252.0 / max(avg_holding_days, 0.5))

    shs = []
    for _ in range(n_sim):
        sample = np.random.choice(trade_returns, n, replace=True)
        mean_r = sample.mean()
        std_r = sample.std()
        if std_r > 1e-10:
            shs.append(mean_r / std_r * ann_factor)

    if not shs:
        return {"median": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0, "n": n}

    shs = np.array(shs)
    alpha = (1 - confidence) / 2
    return {
        "median": float(np.median(shs)),
        "lower": float(np.percentile(shs, alpha * 100)),
        "upper": float(np.percentile(shs, (1 - alpha) * 100)),
        "std": float(shs.std()),
        "n": n,
    }


def bootstrap_pf(
    trade_returns: pd.Series,
    n_sim: int = 10000,
    confidence: float = 0.90,
) -> dict:
    """Bootstrap confidence intervals for Profit Factor from trade returns."""
    n = len(trade_returns)
    if n < 5:
        return {"median": 0.0, "lower": 0.0, "upper": 0.0, "n": n}

    pfs = []
    for _ in range(n_sim):
        sample = np.random.choice(trade_returns, n, replace=True)
        winners = sample[sample > 0].sum()
        losers = abs(sample[sample < 0].sum())
        if losers > 1e-10:
            pfs.append(winners / losers)

    if not pfs:
        return {"median": 0.0, "lower": 0.0, "upper": 0.0, "n": n}

    pfs = np.array(pfs)
    alpha = (1 - confidence) / 2
    return {
        "median": float(np.median(pfs)),
        "lower": float(np.percentile(pfs, alpha * 100)),
        "upper": float(np.percentile(pfs, (1 - alpha) * 100)),
        "n": n,
    }


def compute_deflated_sharpe(sharpe: float, num_trials: int, num_observations: int) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    From Bailey & López de Prado (2014): adjusts observed Sharpe for
    the number of strategy trials attempted, reducing the probability
    that the result is from data mining.

    DSR > 0 means the strategy is statistically significant at the
    5% level after accounting for multiple testing.

    Args:
        sharpe: The observed Sharpe ratio
        num_trials: Number of strategies/configs tried (approximate)
        num_observations: Number of trading days in backtest

    Returns:
        Deflated Sharpe Ratio (DSR). Positive = statistically significant edge.
    """
    from scipy.stats import norm
    if num_observations < 2:
        return 0.0
    # Standard deviation of Sharpe under null (from Mertens' derivation)
    var_sharpe = (1 + 0.5 * sharpe**2) / num_observations
    std_sharpe = np.sqrt(var_sharpe)
    # E[max Z] approximation for num_trials independent trials
    e_max = (1 - np.euler_gamma) * norm.ppf(1 - 1/num_trials) + \
            np.euler_gamma * norm.ppf(1 - 1/(num_trials * np.e))
    dsr = (sharpe - e_max * std_sharpe) / std_sharpe
    return float(dsr)


def generate_report(
    portfolios: dict[str, vbt.Portfolio],
    combined: vbt.Portfolio,
    prop_results: dict[str, dict],
    combined_prop: dict,
    config: dict,
    kelly_factors: dict[str, float] = None,
    bootstrap: bool = True,
) -> dict:
    """Generate performance report with bootstrap CIs and Kelly reporting.

    Args:
        portfolios: Individual instrument portfolios
        combined: Combined portfolio
        prop_results: Prop firm results per instrument
        combined_prop: Combined prop firm results
        config: Backtest config
        kelly_factors: Dict of symbol -> Kelly multiplier (from engine)
        bootstrap: Whether to compute bootstrap confidence intervals
    """
    rows = []
    warnings = []

    # Bootstrap summary header
    if bootstrap:
        print("\n" + "=" * 72)
        print("TRADE BOOTSTRAP CONFIDENCE INTERVALS (90% CI)")
        print("=" * 72)

    for symbol, pf in portfolios.items():
        total_ret = pf.total_return()
        sharpe = _compute_sharpe(pf, config["risk_free_rate"])
        max_dd = pf.max_drawdown()
        trades = pf.trades.count()
        win_rate = pf.trades.win_rate() if trades > 0 else 0.0
        profit_factor = pf.trades.profit_factor() if trades > 0 else 0.0

        if sharpe < 0.5:
            warnings.append(f"WARNING: {symbol} Sharpe ratio < 0.5 ({sharpe:.2f})")
        if max_dd > 0.12:
            warnings.append(f"WARNING: {symbol} max drawdown > 12% ({max_dd*100:.1f}%")
        if win_rate < 0.40:
            warnings.append(f"WARNING: {symbol} win rate < 40% ({win_rate*100:.1f}%)")

        rows.append({
            "Instrument": symbol,
            "Return (%)": total_ret * 100,
            "Sharpe": sharpe,
            "Max DD (%)": max_dd * 100,
            "Trades": trades,
            "Win Rate (%)": win_rate * 100,
            "Profit Factor": profit_factor,
        })

        # Bootstrap diagnostics
        if bootstrap and trades >= 5:
            try:
                rets = pf.trades.records_readable
                if len(rets) > 0:
                    ret_col = [c for c in rets.columns if 'return' in c.lower()]
                    if ret_col:
                        rets_vals = rets[ret_col[0]].values.astype(float)
                        # Compute average holding period in trading days
                        if 'Entry Timestamp' in rets.columns and 'Exit Timestamp' in rets.columns:
                            durations = (rets['Exit Timestamp'] - rets['Entry Timestamp']).dt.total_seconds() / (86400 * 5/7)
                            avg_hold = float(durations.mean()) if len(durations) > 0 else 1.0
                            avg_hold = max(avg_hold, 0.5)  # floor at half a day
                        else:
                            avg_hold = 1.0
                        sh_boot = bootstrap_sharpe(rets_vals, avg_holding_days=avg_hold)
                        pf_boot = bootstrap_pf(rets_vals)
                        print(f"  {symbol:14s}  Sharpe 90% CI: [{sh_boot['lower']:.2f}, {sh_boot['upper']:.2f}]"
                              f"  median={sh_boot['median']:.2f}  PF 90% CI: [{pf_boot['lower']:.2f}, {pf_boot['upper']:.2f}]"
                              f"  n={trades}")
            except Exception:
                pass

    # Combined metrics
    c_total = combined.total_return()
    c_sharpe = _compute_sharpe(combined, config["risk_free_rate"])
    c_max_dd = combined.max_drawdown()
    c_trades = combined.trades.count()
    c_win = combined.trades.win_rate() if c_trades > 0 else 0.0

    if c_max_dd > 0.08:
        warnings.append(f"WARNING: Combined max drawdown > 8% ({c_max_dd*100:.1f}%)")

    c_pf = combined.trades.profit_factor() if c_trades > 0 else 0.0

    rows.append({
        "Instrument": "COMBINED",
        "Return (%)": c_total * 100,
        "Sharpe": c_sharpe,
        "Max DD (%)": c_max_dd * 100,
        "Trades": c_trades,
        "Win Rate (%)": c_win * 100,
        "Profit Factor": c_pf,
    })

    # Bootstraps for combined
    if bootstrap and c_trades >= 5:
        try:
            crets = combined.trades.records_readable
            if len(crets) > 0:
                ret_col = [c for c in crets.columns if 'return' in c.lower()]
                if ret_col:
                    crets_vals = crets[ret_col[0]].values.astype(float)
                    if 'Entry Timestamp' in crets.columns and 'Exit Timestamp' in crets.columns:
                        cdurations = (crets['Exit Timestamp'] - crets['Entry Timestamp']).dt.total_seconds() / (86400 * 5/7)
                        c_avg_hold = float(cdurations.mean()) if len(cdurations) > 0 else 1.0
                        c_avg_hold = max(c_avg_hold, 0.5)
                    else:
                        c_avg_hold = 1.0
                    csh = bootstrap_sharpe(crets_vals, avg_holding_days=c_avg_hold)
                    cpf = bootstrap_pf(crets_vals)
                    print(f"  {'COMBINED':14s}  Sharpe 90% CI: [{csh['lower']:.2f}, {csh['upper']:.2f}]"
                          f"  median={csh['median']:.2f}  PF 90% CI: [{cpf['lower']:.2f}, {cpf['upper']:.2f}]"
                          f"  n={c_trades}")
        except Exception:
            pass

    # Deflated Sharpe
    n_trials = max(len(portfolios), 5)  # approximate number of configs tried
    n_days = c_trades if c_trades > 5 else 252
    dsr = compute_deflated_sharpe(c_sharpe, n_trials, n_days)
    print(f"\n  Deflated Sharpe (trials={n_trials}, obs={n_days}): {dsr:.3f}  "
          f"{'EDGE SIGNIFICANT' if dsr > 0 else 'NOT SIGNIFICANT'}")
    if dsr <= 0:
        warnings.append("WARNING: Deflated Sharpe <= 0 — edge may be from data mining")

    # Kelly factor reporting
    if kelly_factors:
        print("\n" + "=" * 72)
        kelly_frac = RISK_CONFIG.get("kelly_fraction", 0.25)
        label = f"{'Skip' if kelly_frac <= 0 else ''}"
        print(f"  KELLY POSITION SIZING FACTORS ({'Half' if kelly_frac == 0.5 else 'Quarter' if kelly_frac == 0.25 else 'Full' if kelly_frac == 1.0 else str(kelly_frac)})-Kelly")
        print("=" * 72)
        for sym, kf in sorted(kelly_factors.items()):
            if kf <= 0:
                print(f"  {sym:14s}  SKIP (negative edge)")
            else:
                print(f"  {sym:14s}  Kelly mult={kf:.2f}x")
        print("  " + "-" * 60)
        print(f"  {'Combined ATR base':14s}  1.00x (risk_pct=0.002)")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("BACKTEST RESULTS")
    print("=" * 72)
    print(df.to_string(index=False, float_format="%.2f"))
    print("=" * 72)

    for w in warnings:
        print(w)

    # Prop firm results
    print(f"\nProp Firm Simulation (Individual):")
    for sym, res in prop_results.items():
        passed, failure_reason, consistency_triggers, _ = _prop_result(res)
        status = "PASS" if passed else "FAIL"
        print(f"  {sym}: {status}", end="")
        if failure_reason:
            print(f" ({failure_reason})", end="")
        if consistency_triggers > 0:
            print(f" [consistency triggers: {consistency_triggers}]", end="")
        print()

    print(f"\nProp Firm Simulation (Combined):")
    c_passed, c_failure_reason, c_consistency_triggers, c_profit_pct = _prop_result(combined_prop)
    c_status = "PASS" if c_passed else "FAIL"
    print(f"  Combined: {c_status}", end="")
    if c_failure_reason:
        print(f" ({c_failure_reason})", end="")
    print()
    if c_consistency_triggers > 3:
        print(f"  WARNING: Consistency rule triggered {c_consistency_triggers} times")
    print(f"  Trading days: {combined_prop['trading_days_count'] if isinstance(combined_prop, dict) else combined_prop.trading_days_count}")

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)

    return {"summary": df, "warnings": warnings, "prop_results": prop_results, "combined_prop": combined_prop}
