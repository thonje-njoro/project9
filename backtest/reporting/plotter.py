"""Visualization of backtest results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme(style="darkgrid")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_results(
    portfolios: dict,
    combined,
    combined_prop: dict,
) -> None:
    """Generate all result plots."""
    _plot_equity_curves(portfolios)
    _plot_combined(combined, combined_prop)
    _plot_trades(portfolios)


def _plot_equity_curves(portfolios: dict) -> None:
    n = len(portfolios)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    initial = list(portfolios.values())[0].value().iloc[0]

    for i, (symbol, pf) in enumerate(portfolios.items()):
        ax = axes[i]
        equity = pf.value()
        ax.plot(equity.index, equity.values, linewidth=1)
        ax.axhline(y=initial, color="gray", linestyle=":", alpha=0.5, label="Initial capital")
        total_ret = pf.total_return() * 100
        ax.set_title(f"{symbol} — {total_ret:.1f}% return")
        ax.set_ylabel("Equity ($)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)

    if len(portfolios) < len(axes):
        for j in range(len(portfolios), len(axes)):
            axes[j].set_visible(False)

    plt.suptitle("Equity Curves per Instrument", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved equity_curves.png")


def _get_attr(obj, attr, default=None):
    """Get attribute from dict or dataclass."""
    if hasattr(obj, attr):
        return getattr(obj, attr, default)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _plot_combined(combined, combined_prop) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

    equity = combined.value()
    ax1.plot(equity.index, equity.values, linewidth=1, color="steelblue", label="Equity")

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    ax1.fill_between(equity.index, equity.values, rolling_max.values,
                     alpha=0.2, color="red", label="Drawdown")

    initial = equity.iloc[0]
    ax1.axhline(y=initial * (1 - 0.04), color="red", linestyle="--", alpha=0.5, label="-4% DD limit")
    ax1.axhline(y=initial * (1 - 0.10), color="darkred", linestyle="--", alpha=0.5, label="-10% DD limit")

    if _get_attr(combined_prop, "failure_date"):
        fail_date = pd.Timestamp(_get_attr(combined_prop, "failure_date"))
        ax1.axvline(x=fail_date, color="red", linestyle=":", alpha=0.7, label="Prop firm failure")

    ax1.set_title("Combined Portfolio")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30)

    daily_eq = equity.resample("1D").last().dropna()
    daily_pnl = daily_eq.diff()
    colors = ["green" if x >= 0 else "red" for x in daily_pnl.values]
    ax2.bar(daily_pnl.index, daily_pnl.values, color=colors, width=1)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_title("Daily P&L")
    ax2.set_ylabel("P&L ($)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "backtest_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved backtest_results.png")


def _plot_trades(portfolios: dict) -> None:
    n = len(portfolios)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (symbol, pf) in enumerate(portfolios.items()):
        ax = axes[i]
        trades = pf.trades.records_readable
        close = pf.close

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0] if close.shape[1] > 0 else close

        display_end = min(60, len(close))
        close_plot = close.iloc[:display_end]
        ax.plot(close_plot.index, close_plot.values, linewidth=0.8, color="black", alpha=0.7, label="Close")

        if len(trades) > 0 and "Entry Timestamp" in trades.columns:
            long_mask = trades.get("Direction", "") == "Long"
            short_mask = trades.get("Direction", "") == "Short"

            if "Entry Timestamp" in trades.columns:
                entry_times = pd.to_datetime(trades["Entry Timestamp"])
                entry_prices = trades.get("Avg Entry Price", None)

                long_entries = trades[long_mask] if long_mask.any() else pd.DataFrame()
                short_entries = trades[short_mask] if short_mask.any() else pd.DataFrame()

                for _, row in long_entries.iterrows():
                    ts = pd.Timestamp(row["Entry Timestamp"])
                    if ts <= close_plot.index[-1]:
                        ax.scatter(ts, row.get("Avg Entry Price", close_plot.loc[ts] if ts in close_plot.index else np.nan),
                                  marker="^", color="green", s=60, zorder=5)

                for _, row in short_entries.iterrows():
                    ts = pd.Timestamp(row["Entry Timestamp"])
                    if ts <= close_plot.index[-1]:
                        ax.scatter(ts, row.get("Avg Entry Price", close_plot.loc[ts] if ts in close_plot.index else np.nan),
                                  marker="v", color="red", s=60, zorder=5)

                if "Exit Timestamp" in trades.columns:
                    exit_times = pd.to_datetime(trades["Exit Timestamp"])
                    for _, ts in exit_times.items():
                        ts = pd.Timestamp(ts)
                        if ts <= close_plot.index[-1]:
                            price = close_plot.loc[ts] if ts in close_plot.index else np.nan
                            ax.scatter(ts, price, marker="o", color="gray", s=40, zorder=5, alpha=0.7)

        total_ret = pf.total_return() * 100
        ax.set_title(f"{symbol} — {total_ret:.1f}% — First 60 trading periods")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "trades.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved trades.png")
