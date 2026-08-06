"""Parameter sweep engine and full optimization pipeline."""

import json
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path

from config import INSTRUMENTS, RISK_CONFIG, PROP_FIRM_RULES
from risk.position_sizer import compute_atr, atr_position_sizes
from optimization.parameter_grid import get_grid, grid_combinations
from optimization.entry_filters import apply_filters
from optimization.exit_optimizer import ExitOptimizer
from optimization.walk_forward import WalkForwardValidator, detect_overfitting
from optimization.regime_detector import detect_regime, regime_performance_report
from prop_firm.rule_simulator import simulate_prop_firm_rules
from reporting.metrics import _compute_sharpe

MIN_TRADES = 30


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


class StrategyOptimizer:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy: str,
        symbol: str,
        optimize_for: str = "composite_score",
        initial_capital: float = 10_000,
        commission: float = 0.0008,
    ) -> None:
        self.df = df
        self.strategy = strategy
        self.symbol = symbol
        self.optimize_for = optimize_for
        self.initial_capital = initial_capital
        self.commission = commission
        self.freq_map = {"15Min": "15min", "1H": "1h", "4H": "4h"}

    def run(self) -> pd.DataFrame:
        grid = get_grid(self.strategy)
        param_df = grid_combinations(grid)
        close = self.df["close"]
        freq = self.freq_map.get(INSTRUMENTS[self.symbol]["target_tf"], "15min")

        entries_list = []
        exits_list = []

        total = len(param_df)
        for i, (_, row) in enumerate(param_df.iterrows()):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Generating signals: {i+1}/{total}", end="\r")
            entries, exits = self._generate_signals_for_params(row, close)
            entries_list.append(entries)
            exits_list.append(exits)

        print(f"  Generating signals: {total}/{total}")

        entries_matrix = pd.concat(entries_list, axis=1)
        exits_matrix = pd.concat(exits_list, axis=1)

        print(f"  Running vectorbt portfolio backtest...")
        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries_matrix,
                exits=exits_matrix,
                init_cash=self.initial_capital,
                fees=self.commission,
                freq=freq,
            )
        except Exception as e:
            print(f"Portfolio creation failed: {e}")
            return pd.DataFrame()

        trade_counts = pf.trades.count()
        valid_mask = trade_counts >= MIN_TRADES

        results = param_df[valid_mask.values].copy()
        if results.empty:
            print("No parameter combinations met minimum trade count")
            return pd.DataFrame()

        valid_idx = np.where(valid_mask.values)[0]

        results["max_drawdown"] = pf.max_drawdown().values[valid_idx]
        results["win_rate"] = pf.trades.win_rate().values[valid_idx]
        results["profit_factor"] = pf.trades.profit_factor().values[valid_idx]
        results["total_trades"] = pf.trades.count().values[valid_idx]
        results["total_return"] = pf.total_return().values[valid_idx]

        sharpes = []
        rr_ratios = []
        for col_idx in valid_idx:
            try:
                sub_pf = pf[col_idx]
                sharpes.append(_compute_sharpe(sub_pf, 0.045))
                rr_ratios.append(_compute_rr(sub_pf))
            except Exception:
                sharpes.append(0.0)
                rr_ratios.append(0.0)
        results["sharpe"] = sharpes
        results["rr_ratio"] = rr_ratios

        results["composite_score"] = (
            0.35 * results["sharpe"].clip(0, 3) / 3
            + 0.30 * results["rr_ratio"].clip(0, 3) / 3
            + 0.20 * results["win_rate"].clip(0, 1)
            + 0.15 * (1 - results["max_drawdown"].clip(0, 0.2) / 0.2)
        )

        results = results.sort_values(self.optimize_for, ascending=False)
        print(f"  Found {len(results)} valid parameter combinations")
        return results

    def _generate_signals_for_params(self, params: pd.Series, close: pd.Series):
        if self.strategy == "mean_reversion":
            period = int(params["period"])
            threshold = params["std_threshold"]
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            entries = (close < sma - threshold * std).shift(1).fillna(False)
            exits = (close >= sma).shift(1).fillna(False)
        elif self.strategy == "momentum_breakout":
            bp = int(params["breakout_period"])
            high_n = self.df["high"].rolling(bp).max().shift(1)
            low_n = self.df["low"].rolling(bp).min().shift(1)
            vol_mult = params["volume_multiplier"]
            vol_mean = self.df["volume"].rolling(bp).mean()
            vol_ok = self.df["volume"] > vol_mean * vol_mult
            entries = (close > high_n) & vol_ok
            entries = entries.shift(1).fillna(False)
            exits = pd.Series(False, index=close.index)
        elif self.strategy == "trend_following":
            fast = int(params["fast_ema"])
            slow = int(params["slow_ema"])
            if fast >= slow:
                fast, slow = slow, fast
            fe = close.ewm(span=fast, adjust=False).mean()
            se = close.ewm(span=slow, adjust=False).mean()
            entries = ((fe > se) & (fe.shift(1) <= se.shift(1))).shift(1).fillna(False)
            exits = ((fe < se) & (fe.shift(1) >= se.shift(1))).shift(1).fillna(False)
        else:
            entries = pd.Series(False, index=close.index)
            exits = pd.Series(False, index=close.index)

        return entries, exits


def compute_baseline(df: pd.DataFrame, strategy: str, symbol: str) -> dict:
    from config import STRATEGY_PARAMS

    params = STRATEGY_PARAMS[strategy][symbol]
    close = df["close"]

    STRATEGY_ONLY_KEYS = {"trail_atr_mult", "use_vol_calibrated_stop",
                           "vol_stop_base_mult", "vol_stop_min_mult", "vol_stop_max_mult"}
    filtered_params = {k: v for k, v in params.items() if k not in STRATEGY_ONLY_KEYS}

    try:
        if strategy == "mean_reversion":
            from strategies.mean_reversion import generate_signals as gen
            result = gen(df, **filtered_params)
            entries = result[0]
            exits = result[1]
        elif strategy == "momentum_breakout":
            from strategies.momentum_breakout import generate_signals as gen
            result = gen(df, **filtered_params)
            entries = result[0]
            exits = result[1]
        elif strategy == "trend_following":
            from strategies.trend_following import generate_signals as gen
            result = gen(df, **filtered_params)
            entries = result[0]
            exits = result[1]
        else:
            return {"sharpe": 0, "rr_ratio": 0, "win_rate": 0, "max_drawdown": 0, "trades": 0}

        freq_map = {"15Min": "15min", "1H": "1h", "4H": "4h"}
        freq = freq_map.get(INSTRUMENTS[symbol]["target_tf"], "15min")

        pf = vbt.Portfolio.from_signals(
            close=close, entries=entries, exits=exits,
            init_cash=10_000, fees=0.0008, freq=freq,
        )
        trades = pf.trades.count()
        if trades < 5:
            return {"sharpe": 0, "rr_ratio": 0, "win_rate": 0, "max_drawdown": 0, "trades": 0}

        sharpe = _compute_sharpe(pf, 0.045)
        rr = _compute_rr(pf)

        return {
            "sharpe": sharpe,
            "rr_ratio": rr,
            "win_rate": pf.trades.win_rate(),
            "max_drawdown": pf.max_drawdown(),
            "total_return": pf.total_return(),
            "trades": trades,
        }
    except Exception as e:
        print(f"  Baseline error: {e}")
        return {"sharpe": 0, "rr_ratio": 0, "win_rate": 0, "max_drawdown": 0, "trades": 0}


def run_full_optimization(symbol: str, strategy: str, target: str = "composite_score") -> dict:
    print(f"\n{'='*60}")
    print(f"OPTIMIZING: {symbol} ({strategy})")
    print(f"{'='*60}")

    info = INSTRUMENTS[symbol]
    from data.synthetic import generate_synthetic_data
    from data.resampler import resample_ohlcv
    from config import BACKTEST_CONFIG

    raw = generate_synthetic_data({symbol: info}, BACKTEST_CONFIG)
    df = resample_ohlcv(raw[symbol], info["target_tf"], info["asset_class"])

    print(f"\nPhase 1: Baseline")
    baseline = compute_baseline(df, strategy, symbol)
    print(f"  Sharpe: {baseline.get('sharpe', 0):.2f}, RR: {baseline.get('rr_ratio', 0):.2f}, "
          f"Win Rate: {baseline.get('win_rate', 0):.1%}, Max DD: {baseline.get('max_drawdown', 0):.1%}")

    print(f"\nPhase 2: Parameter Sweep")
    opt = StrategyOptimizer(df, strategy, symbol, optimize_for=target)
    results = opt.run()
    if results.empty:
        print("  No valid parameter combinations found")
        return baseline

    top10 = results.head(10)
    print(f"\n  Top 10 parameter sets:")
    for _, row in top10.iterrows():
        params = {k: v for k, v in row.items() if k in ["period", "std_threshold", "breakout_period", "volume_multiplier", "trail_atr_mult", "min_breakout_atr", "fast_ema", "slow_ema", "entry_atr_buffer", "atr_stop_mult"]}
        metrics = f"Sharpe={row['sharpe']:.2f} RR={row['rr_ratio']:.2f} WR={row['win_rate']:.1%} DD={row['max_drawdown']:.1%}"
        print(f"    {params} -> {metrics}")

    print(f"\nPhase 3: Entry Filter Testing")
    best_params = top10.iloc[0].to_dict()

    print(f"\nPhase 4: Exit Optimization")
    close = df["close"]
    entries, exits = opt._generate_signals_for_params(pd.Series(best_params), close)
    exit_opt = ExitOptimizer(df, entries, exits, strategy)
    exit_results = exit_opt.run()
    if not exit_results.empty:
        print(f"  Best exit config: {exit_results.iloc[0]['config']}")
        print(exit_results.to_string(index=False))

    print(f"\nPhase 5: Walk-Forward Validation")
    wf = WalkForwardValidator(df, strategy, symbol)
    wf_result = wf.run(best_params)
    print(f"  OOS Sharpe: {wf_result['oos_sharpe']:.2f}")
    print(f"  Recommendation: {wf_result['recommendation']}")

    overfit = detect_overfitting(wf_result)
    if overfit["overfitting_detected"]:
        for w in overfit["warnings"]:
            print(f"  WARNING: {w}")

    print(f"\nPhase 6: Regime Analysis")
    regimes = detect_regime(df)
    regime_report = regime_performance_report(entries, exits, df, regimes)
    if not regime_report.empty:
        print(regime_report.to_string(index=False))

    print(f"\nPhase 7: Final Report")
    final = {
        "symbol": symbol,
        "strategy": strategy,
        "baseline": baseline,
        "optimized_params": best_params,
        "oos_sharpe": wf_result["oos_sharpe"],
        "oos_recommendation": wf_result["recommendation"],
        "composite_score": best_params.get("composite_score", 0),
    }

    output_dir = Path("results/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    top10.to_csv(output_dir / f"{symbol}_top_params.csv", index=False)

    best_json = {
        symbol: {
            "strategy": strategy,
            **{k: v for k, v in best_params.items() if k != "composite_score"},
            "walk_forward_result": wf_result["recommendation"],
            "oos_sharpe": wf_result["oos_sharpe"],
        }
    }
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_json, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {symbol}")
    print(f"{'='*60}")
    return final
