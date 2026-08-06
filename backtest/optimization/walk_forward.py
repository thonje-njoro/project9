"""Walk-forward validation to prevent overfitting."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from reporting.metrics import _compute_sharpe


def _compute_rr(portfolio) -> float:
    """Compute RR ratio from portfolio trades using vbt 1.0 API."""
    try:
        pnl = np.array(portfolio.trades.pnl.values)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        return float(np.mean(wins) / abs(np.mean(losses)))
    except Exception:
        return 0.0


class WalkForwardValidator:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy: str,
        symbol: str,
        n_windows: int = 4,
        is_pct: float = 0.70,
        optimize_for: str = "composite_score",
    ) -> None:
        self.df = df
        self.strategy = strategy
        self.symbol = symbol
        self.n_windows = n_windows
        self.is_pct = is_pct
        self.optimize_for = optimize_for

    def run(self, best_params: dict) -> dict:
        total_bars = len(self.df)
        window_size = total_bars // self.n_windows

        windows = []
        oos_sharpes = []
        is_sharpes = []

        for i in range(self.n_windows):
            start = i * window_size
            end = min(start + window_size, total_bars)
            if end - start < 50:
                continue

            window_df = self.df.iloc[start:end].copy()
            split = int(len(window_df) * self.is_pct)
            is_df = window_df.iloc[:split]
            oos_df = window_df.iloc[split:]

            if len(is_df) < 30 or len(oos_df) < 10:
                continue

            is_start = str(is_df.index[0].date())
            is_end = str(is_df.index[-1].date())
            oos_start = str(oos_df.index[0].date())
            oos_end = str(oos_df.index[-1].date())

            is_sharpe, is_rr, is_wr = self._run_with_params(is_df, best_params)
            oos_sharpe, oos_rr, oos_wr = self._run_with_params(oos_df, best_params)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

            windows.append({
                "is_period": (is_start, is_end),
                "oos_period": (oos_start, oos_end),
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "is_rr_ratio": is_rr,
                "oos_rr_ratio": oos_rr,
                "is_win_rate": is_wr,
                "oos_win_rate": oos_wr,
            })

        avg_is = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0
        ratio = avg_oos / avg_is if avg_is > 0 else 0

        if ratio >= 0.5 and avg_oos > 0:
            recommendation = "ROBUST"
        elif ratio >= 0.3 and avg_oos > 0:
            recommendation = "MARGINAL"
        else:
            recommendation = "OVERFIT"

        return {
            "windows": windows,
            "oos_combined_return": np.mean([w["oos_sharpe"] for w in windows]) if windows else 0,
            "oos_sharpe": avg_oos,
            "oos_max_drawdown": 0.0,
            "is_to_oos_sharpe_ratio": ratio,
            "recommendation": recommendation,
        }

    def _run_with_params(self, df: pd.DataFrame, params: dict) -> tuple[float, float, float]:
        import vectorbt as vbt

        close = df["close"]
        strategy = self.strategy

        if strategy == "mean_reversion":
            period = int(params.get("period", 20))
            threshold = params.get("std_threshold", 1.5)
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            entries = (close < sma - threshold * std).shift(1).fillna(False)
            exits = (close >= sma).shift(1).fillna(False)
        elif strategy == "momentum_breakout":
            bp = int(params.get("breakout_period", 20))
            high_n = df["high"].rolling(bp).max().shift(1)
            entries = (close > high_n).shift(1).fillna(False)
            exits = pd.Series(False, index=df.index)
        elif strategy == "trend_following":
            fast = int(params.get("fast_ema", 50))
            slow = int(params.get("slow_ema", 200))
            if fast >= slow:
                fast, slow = slow, fast
            fe = close.ewm(span=fast, adjust=False).mean()
            se = close.ewm(span=slow, adjust=False).mean()
            entries = ((fe > se) & (fe.shift(1) <= se.shift(1))).shift(1).fillna(False)
            exits = ((fe < se) & (fe.shift(1) >= se.shift(1))).shift(1).fillna(False)
        else:
            return 0, 0, 0

        try:
            pf = vbt.Portfolio.from_signals(
                close=close, entries=entries, exits=exits,
                init_cash=10_000, fees=0.0008, freq="15min",
            )
            trades = pf.trades.count()
            if trades < 10:
                return 0, 0, 0

            sharpe = _compute_sharpe(pf, 0.045)
            rr = _compute_rr(pf)
            wr = pf.trades.win_rate()
            return sharpe, rr, wr
        except Exception:
            return 0, 0, 0

    def plot(self, output_path: str) -> None:
        pass


def detect_overfitting(wf_results: dict) -> dict:
    warnings = []
    is_sharpes = [w["is_sharpe"] for w in wf_results.get("windows", [])]
    oos_sharpes = [w["oos_sharpe"] for w in wf_results.get("windows", [])]

    if is_sharpes and oos_sharpes:
        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        if avg_is > 0:
            degradation = 1 - (avg_oos / avg_is)
            if degradation > 0.5:
                warnings.append(f"IS/OOS Sharpe degradation: {degradation*100:.0f}%")

    oos_positive = sum(1 for s in oos_sharpes if s > 0)
    if len(oos_sharpes) >= 4 and oos_positive < 3:
        warnings.append(f"OOS Sharpe positive in only {oos_positive}/{len(oos_sharpes)} windows")

    oos_rr = [w["oos_rr_ratio"] for w in wf_results.get("windows", [])]
    if oos_rr and np.mean(oos_rr) < 1.0:
        warnings.append(f"OOS RR ratio below 1.0: {np.mean(oos_rr):.2f}")

    return {
        "overfitting_detected": len(warnings) > 0,
        "warnings": warnings,
    }
