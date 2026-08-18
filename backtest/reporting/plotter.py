"""
reporting/plotter.py — Equity curves, heatmaps → results/
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve",
                      filename: str = "equity_curve.png"):
    """
    Plot and save equity curve.

    Args:
        equity: Equity time series.
        title: Plot title.
        filename: Output filename.
    """
    ensure_results_dir()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, linewidth=1.5, color="#2196F3")
    ax.fill_between(equity.index, equity.values, alpha=0.1, color="#2196F3")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    logger.info(f"Saved equity curve: {filename}")


def plot_drawdown(equity: pd.Series, title: str = "Drawdown",
                  filename: str = "drawdown.png"):
    """Plot drawdown chart."""
    ensure_results_dir()

    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, color="#F44336", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_sharpe_heatmap(results: dict, filename: str = "sharpe_heatmap.png"):
    """
    Plot Sharpe ratio heatmap across instruments.

    Args:
        results: Dict of {symbol: result_dict} with 'sharpe' key.
    """
    ensure_results_dir()

    symbols = list(results.keys())
    sharpes = [results[s].get("sharpe", 0) for s in symbols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(symbols) * 0.5)))

    colors = ["#F44336" if s < 0 else "#4CAF50" if s > 1 else "#FFC107" for s in sharpes]
    bars = ax.barh(symbols, sharpes, color=colors, edgecolor="white")

    ax.set_title("Sharpe Ratio by Instrument", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=1, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Sharpe heatmap: {filename}")


def plot_monte_carlo(mc_results: dict, filename: str = "monte_carlo.png"):
    """
    Plot Monte Carlo Sharpe distribution.

    Args:
        mc_results: Monte Carlo result dict with sharpe values.
    """
    ensure_results_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulated distribution (in production, store the actual bootstrap array)
    median = mc_results.get("sharpe_median", 0)
    pct5 = mc_results.get("sharpe_5pct", 0)
    pct95 = mc_results.get("sharpe_95pct", 0)

    # Generate synthetic distribution for visualization
    rng = np.random.RandomState(42)
    samples = rng.normal(median, (pct95 - pct5) / 4, 2000)

    ax.hist(samples, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=median, color="red", linewidth=2, label=f"Median: {median:.2f}")
    ax.axvline(x=pct5, color="orange", linewidth=1, linestyle="--",
               label=f"5th pct: {pct5:.2f}")
    ax.axvline(x=0, color="black", linewidth=1)

    ax.set_title("Monte Carlo Sharpe Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
