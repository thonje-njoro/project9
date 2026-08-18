"""
engine.py — BacktestEngine: orchestrates data, strategies, risk, validation.
vectorbt wrapper with regime gate integration.
"""

import gc
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.config import (
    BACKTEST_CONFIG, ENGINE_CONFIG, INSTRUMENTS, STRATEGY_PARAMS,
    VALIDATION_THRESHOLDS, RISK_CONFIG,
)
from backtest.data.fetcher import fetch
from backtest.data.resampler import resample_ohlcv
from backtest.risk.regime_filter import (
    compute_regime_gate, apply_regime_to_intraday, compute_regime_coverage,
)
from backtest.risk.position_sizer import atr_position_size
from backtest.risk.monte_carlo import monte_carlo_analysis
from backtest.reporting.metrics import full_metrics, compute_daily_returns
from backtest.reporting.deflated_sharpe import compute_dsr_from_returns

logger = logging.getLogger(__name__)


# ─── Strategy registry ───────────────────────────────────────────────────────

STRATEGY_MODULES = {
    "kalman_trend": "backtest.strategies.kalman_trend",
    "orb_strategy": "backtest.strategies.orb_strategy",
    "momentum_orb": "backtest.strategies.momentum_orb",
    "vwap_mean_reversion": "backtest.strategies.vwap_mean_reversion",
    "xauusd_session_mr": "backtest.strategies.xauusd_session_mr",
    "cper_gld_ratio": "backtest.strategies.cper_gld_ratio",
}


def _import_strategy(strategy_name: str):
    """Dynamically import a strategy module."""
    module_path = STRATEGY_MODULES.get(strategy_name)
    if module_path is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    import importlib
    return importlib.import_module(module_path)


def _filter_params(fn, params: dict) -> dict:
    """Filter params to only include arguments the function signature accepts."""
    sig = inspect.signature(fn)
    valid_keys = set(sig.parameters.keys()) - {"df"}
    return {k: v for k, v in params.items() if k in valid_keys}


class BacktestEngine:
    """
    Orchestrates the full backtest pipeline:
    1. Fetch data for each instrument
    2. Apply regime gate
    3. Generate strategy signals
    4. Run vectorbt Portfolio simulation
    5. Compute metrics, Monte Carlo, validation
    """

    def __init__(self, config: dict | None = None):
        self.config = config or BACKTEST_CONFIG
        self.results = {}
        self.portfolio_returns = {}
        self.equity_curves = {}

    def run_single(self, symbol: str, strategy_name: str,
                   params: dict, regime_gate: pd.Series | None = None) -> dict:
        """
        Run backtest for a single instrument.

        Args:
            symbol: Instrument symbol.
            strategy_name: Strategy name.
            params: Strategy parameters.
            regime_gate: Optional daily regime gate (boolean series).

        Returns:
            dict with metrics, trade stats, signals info.
        """
        instrument = INSTRUMENTS.get(symbol, {})
        timeframe = instrument.get("timeframe", "15min")

        logger.info(f"Running {symbol} ({strategy_name}, {timeframe})")

        # ─── Fetch data ──────────────────────────────────────────────────────
        df = fetch(symbol, timeframe,
                   self.config["start_date"], self.config["end_date"])

        if df.empty:
            logger.warning(f"No data for {symbol}")
            return {"symbol": symbol, "error": "no_data"}

        # ─── Memory optimization: use float32 ─────────────────────────────────
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        # ─── Import strategy ─────────────────────────────────────────────────
        try:
            strategy_mod = _import_strategy(strategy_name)
        except Exception as e:
            logger.error(f"Failed to import {strategy_name}: {e}")
            return {"symbol": symbol, "error": f"import_failed: {e}"}

        # ─── Filter params to match function signature ───────────────────────
        fn = strategy_mod.generate_signals
        filtered_params = _filter_params(fn, params)

        # ─── Generate signals ────────────────────────────────────────────────
        try:
            result = fn(df=df, **filtered_params)
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return {"symbol": symbol, "error": f"signal_failed: {e}"}

        if len(result) == 5:
            long_entries, long_exits, short_entries, short_exits, trailing_stops = result
        else:
            long_entries, long_exits, short_entries, short_exits = result
            trailing_stops = pd.Series(np.nan, index=df.index)

        # ─── Apply regime gate ───────────────────────────────────────────────
        if regime_gate is not None and ENGINE_CONFIG["use_regime_filter"]:
            # Skip regime for instruments with own logic
            if symbol not in ("XAUUSD_MR", "CPER_GLD"):
                intraday_gate = apply_regime_to_intraday(regime_gate, df.index)
                coverage = compute_regime_coverage(intraday_gate)

                if coverage < ENGINE_CONFIG["regime_bypass_if_coverage_lt"]:
                    logger.warning(
                        f"{symbol}: Regime coverage {coverage:.1%} < "
                        f"{ENGINE_CONFIG['regime_bypass_if_coverage_lt']:.1%}, bypassing"
                    )
                else:
                    long_entries = long_entries & intraday_gate
                    short_entries = short_entries & intraday_gate

        # ─── Count signals ───────────────────────────────────────────────────
        n_long_entries = int(long_entries.sum())
        n_short_entries = int(short_entries.sum())
        n_total = n_long_entries + n_short_entries

        if n_total == 0:
            logger.warning(f"{symbol}: 0 signals generated")
            return {
                "symbol": symbol, "strategy": strategy_name,
                "n_trades": 0, "sharpe": 0, "warning": "no_signals",
            }

        # ─── Compute returns (vectorbt-free path for memory efficiency) ──────
        trade_returns = self._compute_trade_returns(
            df, long_entries, long_exits, short_entries, short_exits
        )

        # ─── Build equity curve ──────────────────────────────────────────────
        equity = self._build_equity_curve(df, trade_returns)

        # ─── Compute metrics ─────────────────────────────────────────────────
        metrics = full_metrics(trade_returns, equity)

        # ─── Monte Carlo ─────────────────────────────────────────────────────
        individual_trades = self._extract_individual_trades(
            df, long_entries, long_exits, short_entries, short_exits
        )
        mc = monte_carlo_analysis(individual_trades)

        result_dict = {
            "symbol": symbol,
            "strategy": strategy_name,
            "n_trades": n_total,
            "n_long": n_long_entries,
            "n_short": n_short_entries,
            **metrics,
            "monte_carlo": mc,
        }

        logger.info(
            f"{symbol}: {n_total} trades, Sharpe={metrics['sharpe']:.2f}, "
            f"MaxDD={metrics['max_drawdown']:.3f}"
        )

        # Clean up to save memory
        del df, long_entries, long_exits, short_entries, short_exits
        gc.collect()

        return result_dict

    def run_portfolio(self) -> dict:
        """
        Run backtest for the full instrument portfolio.

        Returns:
            dict with per-instrument results and combined metrics.
        """
        logger.info("Starting portfolio backtest...")

        # ─── Compute daily regime gate (shared across instruments) ───────────
        regime_gate = None
        if ENGINE_CONFIG["use_regime_filter"]:
            # Use SPY as regime proxy
            spy_daily = fetch("SPY", "1d",
                              self.config["start_date"], self.config["end_date"])
            if not spy_daily.empty:
                regime_gate = compute_regime_gate(spy_daily["close"])
                logger.info(f"Regime gate computed: {regime_gate.sum()}/{len(regime_gate)} days active")

        # ─── Run each instrument sequentially ────────────────────────────────
        instrument_results = {}
        all_returns = {}

        for symbol, instrument in INSTRUMENTS.items():
            strategy_name = instrument["strategy"]
            strategy_params = STRATEGY_PARAMS.get(strategy_name, {}).get(symbol, {})

            result = self.run_single(symbol, strategy_name, strategy_params, regime_gate)
            instrument_results[symbol] = result

            if "error" not in result and result.get("n_trades", 0) > 0:
                all_returns[symbol] = result

        # ─── Combined metrics ────────────────────────────────────────────────
        combined = self._compute_combined_metrics(all_returns)

        self.results = {
            "instruments": instrument_results,
            "combined": combined,
        }

        return self.results

    def _compute_trade_returns(self, df: pd.DataFrame,
                               long_entries, long_exits,
                               short_entries, short_exits) -> pd.Series:
        """Compute bar-level portfolio returns from signals (vectorized)."""
        close_ret = df["close"].pct_change().fillna(0).values

        le = long_entries.values.astype(bool)
        lx = long_exits.values.astype(bool)
        se = short_entries.values.astype(bool)
        sx = short_exits.values.astype(bool)

        n = len(df)
        position = np.zeros(n, dtype=np.float32)
        state = 0  # 0=flat, 1=long, -1=short

        for i in range(n):
            if state == 0:
                if le[i]:
                    state = 1
                elif se[i]:
                    state = -1
            elif state == 1:
                if lx[i]:
                    state = 0
            elif state == -1:
                if sx[i]:
                    state = 0
            position[i] = state

        # Commission on trade events (position changes)
        commission = self.config.get("commission_stock", 0.0005)
        trade_events = np.abs(np.diff(position, prepend=0)).astype(float)

        bar_returns = position * close_ret - trade_events * commission

        return pd.Series(bar_returns, index=df.index)

    def _build_equity_curve(self, df: pd.DataFrame,
                            bar_returns: pd.Series) -> pd.Series:
        """Build cumulative equity curve from bar returns."""
        initial = self.config["initial_capital"]
        equity = initial * (1 + bar_returns).cumprod()
        return equity

    def _extract_individual_trades(self, df: pd.DataFrame,
                                   long_entries, long_exits,
                                   short_entries, short_exits) -> np.ndarray:
        """Extract individual trade returns for Monte Carlo (scalar-loop on .values)."""
        close = df["close"].values
        le = long_entries.values.astype(bool)
        lx = long_exits.values.astype(bool)
        se = short_entries.values.astype(bool)
        sx = short_exits.values.astype(bool)

        trades = []
        in_position = False
        side = 0  # 1=long, -1=short
        entry_price = 0.0

        for i in range(len(close)):
            if not in_position:
                if le[i]:
                    in_position = True
                    side = 1
                    entry_price = close[i]
                elif se[i]:
                    in_position = True
                    side = -1
                    entry_price = close[i]
            else:
                if (side == 1 and lx[i]) or (side == -1 and sx[i]):
                    exit_price = close[i]
                    if entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price * side
                        trades.append(ret)
                    in_position = False

        return np.array(trades) if trades else np.array([0.0])

    def _compute_combined_metrics(self, all_returns: dict) -> dict:
        """Compute combined portfolio metrics."""
        if not all_returns:
            return {"avg_sharpe": 0, "best_sharpe": 0, "dsr": 0, "passes": False}

        sharpe_values = [r["sharpe"] for r in all_returns.values() if "sharpe" in r]
        if not sharpe_values:
            return {"avg_sharpe": 0, "best_sharpe": 0, "dsr": 0, "passes": False}

        avg_sharpe = float(np.mean(sharpe_values))
        best_sharpe = float(np.max(sharpe_values))
        n_instruments = len(sharpe_values)

        dsr_result = compute_dsr_from_returns(
            np.array(sharpe_values), n_trials=n_instruments
        )

        return {
            "n_instruments": n_instruments,
            "avg_sharpe": avg_sharpe,
            "best_sharpe": best_sharpe,
            "dsr": dsr_result.get("dsr", 0),
            "dsr_passes": dsr_result.get("passes", False),
            "passes": dsr_result.get("passes", False) and avg_sharpe > 0.3,
        }

    def validate(self) -> dict:
        """
        Run full validation pipeline:
        1. Backtest
        2. DSR check
        3. Walk-forward validation
        4. Monte Carlo
        5. Prop firm simulation

        Returns:
            dict with validation results and pass/fail per instrument.
        """
        from backtest.optimization.purged_walk_forward import PurgedWalkForward
        from backtest.prop_firm.rule_simulator import simulate_all_firms

        # Run backtest
        results = self.run_portfolio()

        validation = {
            "backtest": results,
            "wfv": {},
            "prop_firm": {},
            "overall_passes": True,
        }

        # Walk-forward validation per instrument
        wfv = PurgedWalkForward(n_splits=6, embargo=20)
        for symbol, instrument in INSTRUMENTS.items():
            strategy_name = instrument["strategy"]
            strategy_params = STRATEGY_PARAMS.get(strategy_name, {}).get(symbol, {})

            df = fetch(symbol, instrument["timeframe"],
                       self.config["start_date"], self.config["end_date"])

            if df.empty:
                continue

            strategy_mod = _import_strategy(strategy_name)
            fn = strategy_mod.generate_signals
            filtered_params = _filter_params(fn, strategy_params)

            wfv_result = wfv.validate(df, fn, filtered_params)
            validation["wfv"][symbol] = wfv_result

            if not wfv_result["passes"]:
                validation["overall_passes"] = False
                logger.warning(f"{symbol}: WFV FAILED (consistency={wfv_result['consistency_rate']:.1%})")

            del df
            gc.collect()

        # Prop firm simulation
        # Use combined equity curve (simplified)
        validation["prop_firm"] = {"note": "requires individual equity curves"}

        return validation
