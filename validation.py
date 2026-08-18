"""
validation.py — Monte Carlo + Walk-Forward validation for all project9 symbols.

Drop this in the project9 root. It calls your existing engine.run_backtest()
and works on the trade-level returns it produces.

Usage:
    python validation.py               # runs both WFV + MC on all symbols
    python validation.py --mode wfv    # walk-forward only
    python validation.py --mode mc     # monte carlo only
    python validation.py --symbol SPY  # single symbol
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── adjust these imports to match your actual engine interface ──────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backtest"))

from config import INSTRUMENTS, STRATEGY_PARAMS, ENGINE_CONFIG

# Derive SYMBOLS and STRATEGIES from project9 config
SYMBOLS = list(INSTRUMENTS.keys())
STRATEGIES = list(set(v["strategy"] for v in INSTRUMENTS.values()))

def _build_default_params():
    """Build DEFAULT_PARAMS from STRATEGY_PARAMS config."""
    params = {}
    for strat_name, symbol_params in STRATEGY_PARAMS.items():
        if symbol_params:
            first_symbol = next(iter(symbol_params))
            params[strat_name] = symbol_params[first_symbol]
    return params

DEFAULT_PARAMS = _build_default_params()


class ValidationEngine:
    """Wrapper around project9 engine for validation."""
    
    def __init__(self):
        self.data = {}
    
    def load_data(self, symbol: str, start: str, end: str):
        """Load data for a symbol from cache."""
        from data import fetcher
        df = fetcher.load_cached_data(symbol, start, end)
        if df is not None and not df.empty:
            self.data[symbol] = df
        return df
    
    def run_backtest(self, symbol: str, strategy: str, start: str, end: str, params: dict = None) -> dict:
        """Run backtest for a single symbol/strategy and return trades."""
        import vectorbt as vbt
        
        if symbol not in self.data:
            df = self.load_data(symbol, start, end)
            if df is None:
                return {"trades": pd.DataFrame()}
        else:
            df = self.data[symbol]
        
        # Filter to date range
        mask = (df.index >= start) & (df.index <= end)
        df_period = df[mask].copy()
        
        if len(df_period) < 50:
            return {"trades": pd.DataFrame()}
        
        # Import signal generator
        strategy_info = INSTRUMENTS.get(symbol, {})
        strat_name = strategy_info.get("strategy", strategy)
        
        try:
            result = self._generate_signals(df_period, strat_name, params or {})
        except Exception as e:
            log.warning(f"Signal generation failed for {symbol}/{strat_name}: {e}")
            return {"trades": pd.DataFrame()}
        
        if result is None:
            return {"trades": pd.DataFrame()}
        
        long_entries, long_exits, short_entries, short_exits = result[:4]
        
        # Build portfolio
        pf = vbt.Portfolio.from_signals(
            close=df_period["close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=10000,
            fees=0.001,
            freq="1h",
        )
        
        # Extract trades
        trades_df = pf.trades.records_readable
        
        return {"trades": trades_df, "portfolio": pf}
    
    def _generate_signals(self, df, strategy_name, params):
        """Import and run the signal generator for a strategy."""
        strategy_map = {
            "kalman_trend": "kalman_trend",
            "mean_reversion": "mean_reversion",
            "momentum_breakout": "momentum_breakout",
            "trend_following": "trend_following",
            "crypto_momentum": "crypto_momentum",
            "gold_oil_ratio": "gold_oil_ratio",
            "short_volatility": "short_volatility",
            "cointegration_pair": "cointegration_pair",
            "tsmom": "tsmom",
            "cper_gld_ratio": "cper_gld_ratio",
            "prop_firm_sprint": "prop_firm_sprint",
            "vwap_mean_reversion": "vwap_mean_reversion",
            "orb": "orb_strategy",
            "xauusd_session_mr": "xauusd_session_mr",
            "momentum_orb": "momentum_orb",
        }
        
        module_name = strategy_map.get(strategy_name)
        if not module_name:
            log.warning(f"Unknown strategy: {strategy_name}")
            return None
        
        module = __import__(f"strategies.{module_name}", fromlist=["generate_signals"])
        return module.generate_signals(df, **params)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validation.log"),
    ],
)
log = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────────────

WFV_CONFIG = {
    "train_months": 6,       # in-sample window
    "test_months": 1,        # out-of-sample step
    "min_trades": 20,        # skip OOS window if fewer trades (not enough signal)
    "anchored": False,       # True = expanding window; False = rolling window
}

MC_CONFIG = {
    "n_sims": 2_000,
    "confidence_levels": [0.05, 0.25, 0.50, 0.75, 0.95],
    "starting_equity": 10_000.0,
    "resample_unit": "trades",  # "trades" | "daily_returns" — trades is more conservative
}

RESULTS_DIR = Path("validation_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class OOSWindow:
    symbol: str
    strategy: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_trades: int
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float


@dataclass
class MCResult:
    symbol: str
    strategy: str
    n_sims: int
    median_return: float
    p5_return: float
    p95_return: float
    median_sharpe: float
    p5_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    prob_positive: float          # P(final equity > starting)
    prob_ruin: float              # P(drawdown > 50%)


# ── core metric helpers ──────────────────────────────────────────────────────

def compute_equity_curve(trade_returns: np.ndarray, start_equity: float = 10_000.0) -> np.ndarray:
    return start_equity * np.cumprod(1 + trade_returns)


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def compute_max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def trades_to_returns(trades_df: pd.DataFrame) -> np.ndarray:
    """
    Extract per-trade return series from backtest output.
    Assumes trades_df has a 'pnl_pct' or 'return' column; adapt to your schema.
    """
    for col in ("pnl_pct", "return", "pnl_percent", "trade_return"):
        if col in trades_df.columns:
            return trades_df[col].dropna().to_numpy(dtype=float)
    # fallback: compute from entry/exit price if present
    if {"entry_price", "exit_price"}.issubset(trades_df.columns):
        return ((trades_df["exit_price"] - trades_df["entry_price"]) / trades_df["entry_price"]).to_numpy()
    raise ValueError(f"Can't find return column in trades. Columns: {list(trades_df.columns)}")


# ── walk-forward validation ──────────────────────────────────────────────────

def generate_wfv_windows(
    full_start: str,
    full_end: str,
    train_months: int,
    test_months: int,
    anchored: bool,
) -> list[tuple[str, str, str, str]]:
    """Returns list of (train_start, train_end, test_start, test_end)."""
    windows = []
    start = pd.Timestamp(full_start)
    end = pd.Timestamp(full_end)

    anchor = start
    train_start = start

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

        if anchored:
            train_start = anchor          # expanding: anchor stays fixed
        else:
            train_start = test_start      # rolling: move forward by test period

    return windows


def run_wfv_for_symbol(
    symbol: str,
    strategy: str,
    full_start: str,
    full_end: str,
    engine: BacktestEngine,
    params: dict,
) -> list[OOSWindow]:
    windows = generate_wfv_windows(
        full_start, full_end,
        WFV_CONFIG["train_months"],
        WFV_CONFIG["test_months"],
        WFV_CONFIG["anchored"],
    )
    if not windows:
        log.warning(f"{symbol}/{strategy}: not enough data for WFV windows")
        return []

    oos_results = []

    for train_start, train_end, test_start, test_end in windows:
        try:
            # ── in-sample: optimise / validate params (placeholder) ──────────
            # If you have a param optimiser, call it here on [train_start, train_end].
            # For now we use the config params as-is — swap in your optimiser output.
            optimised_params = params.copy()

            # ── out-of-sample: evaluate on held-out window ───────────────────
            result = engine.run_backtest(
                symbol=symbol,
                strategy=strategy,
                start=test_start,
                end=test_end,
                params=optimised_params,
            )

            # `result` shape depends on your engine — adapt as needed
            trades = result.get("trades", pd.DataFrame())
            if isinstance(trades, pd.DataFrame) and len(trades) < WFV_CONFIG["min_trades"]:
                log.debug(f"{symbol}/{strategy} {test_start}→{test_end}: only {len(trades)} trades, skipping window")
                continue

            rets = trades_to_returns(trades)

            oos_results.append(OOSWindow(
                symbol=symbol,
                strategy=strategy,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_trades=len(rets),
                sharpe=compute_sharpe(rets),
                total_return=float(np.prod(1 + rets) - 1),
                max_drawdown=compute_max_drawdown(compute_equity_curve(rets)),
                win_rate=float((rets > 0).mean()),
            ))

        except Exception as e:
            log.warning(f"WFV window failed {symbol}/{strategy} {test_start}→{test_end}: {e}")
            continue

    return oos_results


# ── monte carlo ──────────────────────────────────────────────────────────────

def run_mc_for_returns(
    symbol: str,
    strategy: str,
    trade_returns: np.ndarray,
    n_sims: int = MC_CONFIG["n_sims"],
    start_equity: float = MC_CONFIG["starting_equity"],
    confidence_levels: list[float] = MC_CONFIG["confidence_levels"],
) -> MCResult | None:
    if len(trade_returns) < 10:
        log.warning(f"{symbol}/{strategy}: too few trades ({len(trade_returns)}) for MC")
        return None

    n_trades = len(trade_returns)
    rng = np.random.default_rng(seed=42)

    # shape: (n_sims, n_trades) — resample with replacement
    simulated = rng.choice(trade_returns, size=(n_sims, n_trades), replace=True)

    # equity curves: (n_sims, n_trades)
    equity = np.cumprod(1 + simulated, axis=1) * start_equity

    final_equity = equity[:, -1]
    final_returns = (final_equity / start_equity) - 1

    # per-sim Sharpe and max-DD
    means = simulated.mean(axis=1)
    stds = simulated.std(axis=1)
    sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = ((equity - peaks) / peaks).min(axis=1)

    conf = np.percentile(final_returns, [q * 100 for q in confidence_levels])
    conf_map = dict(zip(confidence_levels, conf))

    return MCResult(
        symbol=symbol,
        strategy=strategy,
        n_sims=n_sims,
        median_return=float(np.median(final_returns)),
        p5_return=float(conf_map[0.05]),
        p95_return=float(conf_map[0.95]),
        median_sharpe=float(np.median(sharpes)),
        p5_sharpe=float(np.percentile(sharpes, 5)),
        median_max_dd=float(np.median(drawdowns)),
        p95_max_dd=float(np.percentile(drawdowns, 95)),  # 95th worst drawdown
        prob_positive=float((final_returns > 0).mean()),
        prob_ruin=float((drawdowns < -0.50).mean()),
    )


# ── main loop ────────────────────────────────────────────────────────────────

def run_all(
    symbols: list[str],
    strategies: list[str],
    full_start: str,
    full_end: str,
    mode: str = "both",
) -> dict:
    # Load data for all symbols
    from data import fetcher
    from config import ENGINE_CONFIG
    
    data = {}
    for symbol in symbols:
        try:
            df = fetcher.load_cached_data(symbol, full_start, full_end)
            if df is not None and not df.empty:
                data[symbol] = df
                log.info(f"Loaded {symbol}: {len(df)} bars")
        except Exception as e:
            log.warning(f"Could not load {symbol}: {e}")
    
    if not data:
        log.error("No data loaded. Check data/cache/ directory.")
        return {}
    
    engine = BacktestEngine(
        data=data,
        config=ENGINE_CONFIG,
        use_regime_filter=ENGINE_CONFIG.get("use_regime_filter", True),
    )

    wfv_rows: list[dict] = []
    mc_rows: list[dict] = []

    pairs = [(s, strat) for s in symbols for strat in strategies]

    for symbol, strategy in tqdm(pairs, desc="symbol/strategy pairs"):
        params = DEFAULT_PARAMS.get(strategy, {})

        log.info(f"▶ {symbol} / {strategy}")

        # ── get full-period trades for MC ────────────────────────────────────
        trade_returns = None
        if mode in ("mc", "both"):
            try:
                result = engine.run_backtest(
                    symbol=symbol,
                    strategy=strategy,
                    start=full_start,
                    end=full_end,
                    params=params,
                )
                trades = result.get("trades", pd.DataFrame())
                if not trades.empty:
                    trade_returns = trades_to_returns(trades)
            except Exception as e:
                log.warning(f"Full backtest failed for {symbol}/{strategy}: {e}")

        # ── monte carlo ──────────────────────────────────────────────────────
        if mode in ("mc", "both") and trade_returns is not None:
            mc = run_mc_for_returns(symbol, strategy, trade_returns)
            if mc:
                row = asdict(mc)
                mc_rows.append(row)
                log.info(
                    f"  MC  median_ret={mc.median_return:.1%}  "
                    f"p5={mc.p5_return:.1%}  p95={mc.p95_return:.1%}  "
                    f"P(ruin)={mc.prob_ruin:.1%}"
                )

        # ── walk-forward ─────────────────────────────────────────────────────
        if mode in ("wfv", "both"):
            windows = run_wfv_for_symbol(
                symbol, strategy, full_start, full_end, engine, params
            )
            for w in windows:
                wfv_rows.append(asdict(w))

            if windows:
                sharpes = [w.sharpe for w in windows]
                log.info(
                    f"  WFV {len(windows)} windows  "
                    f"median_sharpe={np.median(sharpes):.2f}  "
                    f"positive_windows={sum(s > 0 for s in sharpes)}/{len(sharpes)}"
                )

    # ── persist results ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output: dict[str, pd.DataFrame | None] = {}

    if wfv_rows:
        wfv_df = pd.DataFrame(wfv_rows)
        wfv_path = RESULTS_DIR / f"wfv_{ts}.csv"
        wfv_df.to_csv(wfv_path, index=False)
        log.info(f"WFV results → {wfv_path}")
        output["wfv"] = wfv_df

        # summary: per symbol/strategy across all OOS windows
        summary = (
            wfv_df.groupby(["symbol", "strategy"])
            .agg(
                n_windows=("sharpe", "count"),
                median_sharpe=("sharpe", "median"),
                pct_positive=("sharpe", lambda x: (x > 0).mean()),
                median_return=("total_return", "median"),
                median_drawdown=("max_drawdown", "median"),
                median_win_rate=("win_rate", "median"),
            )
            .reset_index()
            .sort_values("median_sharpe", ascending=False)
        )
        summary_path = RESULTS_DIR / f"wfv_summary_{ts}.csv"
        summary.to_csv(summary_path, index=False)
        log.info(f"\nWFV SUMMARY:\n{summary.to_string(index=False)}")

    if mc_rows:
        mc_df = pd.DataFrame(mc_rows)
        mc_path = RESULTS_DIR / f"mc_{ts}.csv"
        mc_df.to_csv(mc_path, index=False)
        log.info(f"MC results → {mc_path}")
        output["mc"] = mc_df

        log.info(f"\nMC SUMMARY:\n{mc_df[['symbol','strategy','median_return','p5_return','p95_return','prob_ruin']].sort_values('median_return', ascending=False).to_string(index=False)}")

    return output


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["wfv", "mc", "both"], default="both")
    parser.add_argument("--symbol", help="Run on single symbol only")
    parser.add_argument("--strategy", help="Run on single strategy only")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS
    strategies = [args.strategy] if args.strategy else STRATEGIES

    run_all(
        symbols=symbols,
        strategies=strategies,
        full_start=args.start,
        full_end=args.end,
        mode=args.mode,
    )
