"""Entry point — runs full backtest suite with all 7 improvements.

Item 7: Cointegration pair spread construction in _build_synthetic_instruments()
Item 1-6: Wired through engine and reporting pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from dotenv import load_dotenv
from itertools import product

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BACKTEST_CONFIG, INSTRUMENTS, STRATEGY_PARAMS,
    RISK_CONFIG, PROP_FIRM_RULES, ENGINE_CONFIG, SYSTEM_HEALTH_CONFIG,
    DATA_VENDOR_CONFIG, REFLECTION_CONFIG,
)
from data.fetcher import DataFetcher
from data.resampler import resample_ohlcv
from data.synthetic import generate_synthetic_data
from data.symbol_map import yfinance_symbol, normalize_symbol
from data.vendor_router import get_ohlcv, DEFAULT_VENDOR_CHAINS
from engine import BacktestEngine
from prop_firm.rule_simulator import simulate_prop_firm_rules
from reporting.metrics import generate_report
from reporting.plotter import plot_results
from risk.position_sizer import compute_atr, atr_position_sizes
from risk.system_health_monitor import SystemHealthMonitor
from analysis.trade_memory import log_trade, resolve_trade, load_entries, get_context


def _get_data() -> dict[str, pd.DataFrame]:
    """Fetch data using the vendor router chain (Alpaca → yfinance → Alpha Vantage)."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    use_synthetic = (not api_key or api_key == "your_free_tier_key"
                     or not secret_key or secret_key == "your_free_tier_secret")
    if use_synthetic:
        print("No valid Alpaca API keys found — using synthetic data.\n")
        return generate_synthetic_data(INSTRUMENTS, BACKTEST_CONFIG)
    else:
        real_instruments = {
            sym: info for sym, info in INSTRUMENTS.items()
            if info["asset_class"] != "synthetic"
        }
        alpaca_instruments = {sym: info for sym, info in real_instruments.items()
                              if info["asset_class"] in ("stock", "crypto")}
        forex_instruments = {sym: info for sym, info in real_instruments.items()
                             if info["asset_class"] == "forex"}

        data = {}

        # Synthetic instruments to exclude from Alpaca fetch (derived locally)
        _SYNTHETIC_NAMES = {"IWM_VWAP", "SPY_MOM", "SPY_VOL"}

        # Fetch Alpaca instruments
        if alpaca_instruments:
            # Exclude derived/synthetic instruments from Alpaca fetch
            fetch_instruments = {k: v for k, v in alpaca_instruments.items() if k not in _SYNTHETIC_NAMES}
            fetcher = DataFetcher(api_key, secret_key)
            data.update(fetcher.fetch_all(fetch_instruments, BACKTEST_CONFIG))

            # Vendor-router fallback: any stock/crypto that returned no data
            stock_chain = DATA_VENDOR_CONFIG.get("core_stock_apis",
                                                  DEFAULT_VENDOR_CHAINS["core_stock_apis"])
            for sym in list(alpaca_instruments):
                if sym not in data or data[sym] is None or data[sym].empty:
                    yf_sym = yfinance_symbol(sym)
                    print(f"  {sym} missing from Alpaca; trying vendor chain {stock_chain}...")
                    df = _fetch_with_vendor_router(yf_sym, stock_chain)
                    if df is not None:
                        print(f"  {sym}: {len(df)} bars loaded via vendor router")
                        data[sym] = df

        # Fetch forex instruments via vendor router (yfinance → Alpha Vantage)
        forex_chain = DATA_VENDOR_CONFIG.get("forex_data",
                                              DEFAULT_VENDOR_CHAINS["forex_data"])
        for symbol, _ in forex_instruments.items():
            yf_sym = yfinance_symbol(symbol)
            print(f"Loading {symbol} via vendor chain {forex_chain} ({yf_sym})...")
            df = _fetch_with_vendor_router(yf_sym, forex_chain)
            if df is not None and len(df) > 0:
                data[symbol] = df
                print(f"  {symbol}: {len(df)} bars loaded")
            else:
                print(f"  WARNING: {symbol} returned no data from any vendor, skipping")

        # Derive IWM_VWAP from IWM 1Min data if IWM was fetched
        if "IWM_VWAP" in INSTRUMENTS and "IWM" in data and data["IWM"] is not None:
            iwm_raw = data["IWM"]
            if len(iwm_raw) > 1000:
                iwm_vwap_df = resample_ohlcv(iwm_raw, "15Min", "stock")
                if len(iwm_vwap_df) > 100:
                    data["IWM_VWAP"] = iwm_vwap_df
                    print(f"  IWM_VWAP: {len(iwm_vwap_df)} bars (derived from IWM 1Min @ 15Min)")

        # Derive SPY_MOM (momentum breakout from SPY 1Min @ 15Min)
        if "SPY_MOM" in INSTRUMENTS and "SPY" in data and data["SPY"] is not None:
            spy_raw = data["SPY"]
            if len(spy_raw) > 1000:
                spy_mom_df = resample_ohlcv(spy_raw, "15Min", "stock")
                if len(spy_mom_df) > 100:
                    data["SPY_MOM"] = spy_mom_df
                    print(f"  SPY_MOM: {len(spy_mom_df)} bars (derived from SPY 1Min @ 15Min)")

        # Derive SPY_VOL (short vol from SPY 1Min @ 1D)
        if "SPY_VOL" in INSTRUMENTS and "SPY" in data and data["SPY"] is not None:
            spy_raw = data["SPY"]
            if len(spy_raw) > 1000:
                spy_vol_df = resample_ohlcv(spy_raw, "1D", "stock")
                if len(spy_vol_df) > 100:
                    data["SPY_VOL"] = spy_vol_df
                    print(f"  SPY_VOL: {len(spy_vol_df)} daily bars (derived from SPY 1Min)")

        return data


def _fetch_with_vendor_router(ticker: str, vendor_chain: list[str]) -> pd.DataFrame | None:
    """Fetch data via the vendor router, falling back through the chain.

    Respects the REFLECTION_CONFIG.enabled flag — when disabled, skip Alpha
    Vantage to conserve rate limit.
    """
    from data.vendor_router import get_ohlcv

    # Filter Alpha Vantage out of the chain if reflections/trade-memory are
    # disabled, to conserve the 500-call/day free-tier budget.
    if not REFLECTION_CONFIG.get("enabled", False):
        vendor_chain = [v for v in vendor_chain if v != "alpha_vantage"]

    return get_ohlcv(ticker, BACKTEST_CONFIG["start_date"],
                     BACKTEST_CONFIG["end_date"], vendor_chain=vendor_chain)


# _yfinance_symbol is superseded by data.symbol_map.yfinance_symbol()
# which handles 40+ symbols (metals, energy, forex, crypto, index CFDs).
# Keep this stub for any external code that imports it from main:
def _yfinance_symbol(symbol: str) -> str:
    from data.symbol_map import yfinance_symbol
    return yfinance_symbol(symbol)


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV data from yfinance and flatten to our format."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            return None
        # Flatten yfinance's MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            price_types = {"open", "high", "low", "close", "adj close", "volume"}
            col_names = []
            for col in df.columns:
                name = None
                for level in col:
                    if isinstance(level, str) and level.lower() in price_types:
                        name = level.lower()
                        break
                col_names.append(name or str(col[-1]).lower())
            df.columns = col_names
        else:
            df.columns = [c.lower() for c in df.columns]

        if "adj close" in df.columns and "close" in df.columns:
            df = df.drop(columns=["adj close"])
        elif "adj close" in df.columns:
            df = df.rename(columns={"adj close": "close"})

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            print(f"  Missing columns: {missing}")
            return None

        df = df[["open", "high", "low", "close", "volume"]]

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()
        return df
    except ImportError:
        print("  WARNING: yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"  WARNING: yfinance error for {ticker}: {e}")
        return None


def _build_synthetic_instruments(resampled: dict[str, pd.DataFrame],
                                 raw_data: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Build synthetic instruments: pairs, ratios, cointegration spreads, derived strategies.

    Item 7: Cointegration pair spreads from existing instruments.
    When raw_data is provided, derive VWAP MR instruments (e.g. IWM_VWAP) at 15Min.
    """
    out = dict(resampled)

    # 1. Gold/Oil ratio (existing)
    if "GLD" in resampled and "USO" in resampled and "GLD_USO_RATIO" in INSTRUMENTS:
        gld = resampled["GLD"]
        uso = resampled["USO"]
        common_idx = gld.index.intersection(uso.index)
        ratio_close = gld.loc[common_idx, "close"] / uso.loc[common_idx, "close"]
        ratio_df = pd.DataFrame({
            "open": ratio_close, "high": ratio_close * 1.001,
            "low": ratio_close * 0.999, "close": ratio_close,
            "volume": pd.Series(0, index=common_idx),
        }, index=common_idx)
        out["GLD_USO_RATIO"] = ratio_df
        print(f"  GLD_USO_RATIO: {len(ratio_df)} bars (GLD/USO ratio)")

    # 2. CPER/GLD ratio (copper/gold pair, mean-reverts)
    if "CPER" in resampled and "GLD" in resampled and "CPER_GLD_RATIO" in INSTRUMENTS:
        cper = resampled["CPER"]
        gld = resampled["GLD"]
        common_idx = cper.index.intersection(gld.index)
        ratio_close = cper.loc[common_idx, "close"] / gld.loc[common_idx, "close"]
        ratio_df = pd.DataFrame({
            "open": ratio_close, "high": ratio_close * 1.001,
            "low": ratio_close * 0.999, "close": ratio_close,
            "volume": pd.Series(0, index=common_idx),
        }, index=common_idx)
        out["CPER_GLD_RATIO"] = ratio_df
        print(f"  CPER_GLD_RATIO: {len(ratio_df)} bars (CPER/GLD ratio)")

    # 3. Cointegration pair spreads (Item 7)
    # Find all cointegration_pair instruments
    for sym, info in INSTRUMENTS.items():
        if info.get("strategy") == "cointegration_pair":
            src_sym = info["pair_src"]
            hedge_sym = info["pair_hedge"]
            hedge_tf = info["pair_hedge_tf"]

            if src_sym not in out:
                print(f"  WARNING: {sym} source {src_sym} not available, skipping")
                continue
            if hedge_sym not in out:
                print(f"  WARNING: {sym} hedge {hedge_sym} not available, skipping")
                continue

            # Get source and hedge data (both already resampled)
            src = out[src_sym]
            hedge = out[hedge_sym]

            # Align on common index — use date-only for cross-timeframe pairs
            src_dates = src.index.normalize().unique()
            hedge_dates = hedge.index.normalize().unique()
            common_dates = np.intersect1d(src_dates, hedge_dates)

            if len(common_dates) < 10:
                print(f"  WARNING: {sym} only {len(common_dates)} common dates, skipping")
                continue

            # Map each bar's date to get the hedge price
            hedge_by_date = hedge.groupby(hedge.index.normalize())["close"].last()
            src_close = src["close"].copy()
            src_date = src.index.normalize()
            hedge_aligned = src_date.map(hedge_by_date)

            valid = hedge_aligned.notna() & src_close.notna()
            if valid.sum() < 10:
                print(f"  WARNING: {sym} only {valid.sum()} aligned bars, skipping")
                continue

            spread = src_close[valid] / hedge_aligned[valid]
            spread_df = pd.DataFrame({
                "open": spread, "high": spread * 1.001,
                "low": spread * 0.999, "close": spread,
                "volume": pd.Series(0, index=spread.index),
            }, index=spread.index)
            out[sym] = spread_df
            print(f"  {sym}: {len(spread_df)} bars ({src_sym}/{hedge_sym} cross-tf spread)")

    return out


def main() -> None:
    print("=== BACKTEST ENGINE v2 (Improvements 1-7) ===")
    print(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}")
    print(f"Initial capital: ${BACKTEST_CONFIG['initial_capital']:,}")
    n_states = ENGINE_CONFIG.get("hmm_states", 2)
    use_garch = ENGINE_CONFIG.get("use_garch", False)
    regime_label = f"{n_states}-state HMM{' + GARCH' if use_garch else ''}" if ENGINE_CONFIG.get('use_regime_filter') else 'off'
    print(f"Regime filter: {regime_label} | Kelly sizing: {'on' if RISK_CONFIG.get('use_kelly_sizing') else 'off'} "
          f"| Risk parity: {'on' if ENGINE_CONFIG.get('use_risk_parity') else 'off'}")
    print()

    raw_data = _get_data()

    resampled = {
        sym: resample_ohlcv(df, INSTRUMENTS[sym]["target_tf"], INSTRUMENTS[sym]["asset_class"])
        for sym, df in raw_data.items()
    }
    print(f"\nResampled data:")
    for sym, df in resampled.items():
        print(f"  {sym}: {len(df)} bars ({INSTRUMENTS[sym]['target_tf']})")

    # Build synthetic instruments (Item 7: cointegration pairs)
    resampled = _build_synthetic_instruments(resampled, raw_data)

    engine = BacktestEngine(resampled, BACKTEST_CONFIG,
                            use_regime_filter=ENGINE_CONFIG.get("use_regime_filter", True))
    print("\nRunning individual backtests...")
    portfolios = engine.run()
    print("\nRunning combined portfolio backtest...")
    combined = engine.run_combined()

    print("\nSimulating prop firm rules...")
    prop_results = {
        sym: simulate_prop_firm_rules(pf, PROP_FIRM_RULES)
        for sym, pf in portfolios.items()
    }
    combined_prop = simulate_prop_firm_rules(combined, PROP_FIRM_RULES)

    # Pass Kelly factors to report (Item 1)
    kelly_factors = getattr(engine, 'kelly_factors', None)

    report = generate_report(portfolios, combined, prop_results, combined_prop,
                             BACKTEST_CONFIG, kelly_factors=kelly_factors, bootstrap=True)
    plot_results(portfolios, combined, combined_prop)

    print("\nRunning walk-forward validation...")
    wf_results = {}
    for sym, df in resampled.items():
        wf = _walk_forward_validate(sym, df)
        wf_results[sym] = wf
    print(f"  {'Instrument':14s}  {'Folds':6s}  {'IS PF':10s}  {'OOS PF':10s}  {'PF Decay':8s}  {'Verdict':15s}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*15}")
    for sym, r in sorted(wf_results.items()):
        if r["folds"] > 0:
            print(f"  {sym:14s}  {r['folds']:6d}  {r['avg_is_pf']:10.3f}  {r['avg_oos_pf']:10.3f}  {r['pf_decay']:8.3f}  {r['recommendation']:15s}")

    print("\nSystem health checks...")
    monitor = SystemHealthMonitor(
        historical_max_dd=SYSTEM_HEALTH_CONFIG["historical_max_dd"],
        sharpe_trigger=SYSTEM_HEALTH_CONFIG["sharpe_trigger"],
        profit_factor_trigger=SYSTEM_HEALTH_CONFIG["profit_factor_trigger"],
        lookback_months=SYSTEM_HEALTH_CONFIG["lookback_months"],
        cusum_h=SYSTEM_HEALTH_CONFIG["cusum_h"],
    )
    equity = combined.value()
    health = monitor.full_check(equity)
    if health.healthy:
        print(f"  System health: PASS — {health.metrics}")
    else:
        print(f"  System health: FAIL — {health.breach_reason} (date: {health.breach_date})")

    print("\nResults saved to results/")

    # ── Trade memory: log resolved results for each instrument ──
    if REFLECTION_CONFIG.get("enabled", False):
        _log_trade_memory(portfolios, resampled)


def _log_trade_memory(
    portfolios: dict[str, vbt.Portfolio],
    resampled: dict[str, pd.DataFrame],
) -> None:
    """Log backtest results to the trade memory markdown log.

    Each instrument's trades are summarized as a resolved entry.
    Also resolves any pending entries from prior runs.
    """
    print("\nUpdating trade memory log...")

    # Resolve pending entries (from prior runs that didn't have outcome data)
    pending = [e for e in load_entries() if e.get("pending")]
    if pending:
        print(f"  Resolving {len(pending)} pending entries...")

    for sym, pf in portfolios.items():
        try:
            total_ret = pf.total_return()
            sharpe = pf.sharpe_ratio()
            trades = pf.trades
            trade_count = len(trades) if trades is not None else 0
        except Exception:
            continue

        direction = "Buy" if total_ret > 0 else "Sell"
        # Pick the midpoint of the backtest as a representative date
        df = resampled.get(sym)
        if df is not None and len(df) > 0:
            mid_idx = len(df) // 2
            trade_date = str(df.index[mid_idx].date()) if hasattr(df.index[mid_idx], 'date') else "unknown"
        else:
            trade_date = "unknown"

        log_trade(
            ticker=sym,
            trade_date=trade_date,
            direction=direction,
            entry_price=0.0,
            regime="backtest",
            strategy=INSTRUMENTS.get(sym, {}).get("strategy", "unknown"),
            notes=f"Return={total_ret:+.2%}, Sharpe={sharpe:.2f}, trades={trade_count}",
        )

        # For entries with actual outcome data, resolve them
        if total_ret != 0:
            resolve_trade(
                ticker=sym,
                trade_date=trade_date,
                exit_price=0.0,
                holding_days=5,
                raw_return=total_ret if isinstance(total_ret, (int, float)) else 0.0,
                alpha_return=total_ret if isinstance(total_ret, (int, float)) else 0.0,
                benchmark="SPY",
                reflection="",
            )

    # Print context summary for quick reference
    context = get_context("GLD", n_same=3, n_cross=2)
    if context:
        print(f"Trade memory context:\n{context}")


def parameter_sweep() -> None:
    """Run parameter sweep over key strategy parameters.

    For each instrument's strategy, varies the primary parameter(s) and
    reports the impact on Sharpe, return, and max DD. Results saved to CSV.
    """
    print("=== PARAMETER SWEEP ===\n")
    raw_data = _get_data()
    if not raw_data:
        print("ERROR: No data available for sweep.\n")
        return

    resampled = {
        sym: resample_ohlcv(df, INSTRUMENTS[sym]["target_tf"], INSTRUMENTS[sym]["asset_class"])
        for sym, df in raw_data.items()
    }
    resampled = _build_synthetic_instruments(resampled)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    # 1) Kalman trend: sweep Q (process noise)
    print("--- Kalman Trend: Sweeping Q (process noise) ---")
    kalman_syms = [s for s, v in INSTRUMENTS.items() if v["strategy"] == "kalman_trend"]
    q_values = [0.005, 0.01, 0.02, 0.05, 0.10]
    kalman_rows = []
    for sym in kalman_syms:
        if sym not in resampled:
            continue
        df = resampled[sym]
        for q in q_values:
            engine = BacktestEngine({sym: df}, BACKTEST_CONFIG, use_regime_filter=False)
            engine.kelly_factors = {sym: 1.0}
            try:
                pf = engine._run_single(sym, df, kelly_mult=1.0)
                sharpe = _compute_sharpe(pf, BACKTEST_CONFIG["risk_free_rate"])
                ret = pf.total_return() * 100
                dd = pf.max_drawdown() * 100
            except Exception as e:
                sharpe, ret, dd = 0.0, 0.0, 0.0
            kalman_rows.append({
                "Instrument": sym, "Q": q,
                "Sharpe": f"{sharpe:.2f}", "Return%": f"{ret:.2f}", "MaxDD%": f"{dd:.2f}",
            })
    if kalman_rows:
        kdf = pd.DataFrame(kalman_rows)
        print(kdf.to_string(index=False))
        kdf.to_csv(out_dir / "sweep_kalman_q.csv", index=False)
        print()

    # 2) Momentum breakout: sweep breakout_period
    print("--- Momentum Breakout: Sweeping breakout_period ---")
    mb_syms = [s for s, v in INSTRUMENTS.items() if v["strategy"] == "momentum_breakout"]
    bp_values = [5, 10, 15, 20, 30]
    mb_rows = []
    for sym in mb_syms:
        if sym not in resampled:
            continue
        df = resampled[sym]
        for bp in bp_values:
            # Patch params temporarily
            from config import STRATEGY_PARAMS as sp
            orig = sp.get("momentum_breakout", {}).get(sym, {}).copy()
            sp.setdefault("momentum_breakout", {}).setdefault(sym, {})["breakout_period"] = bp
            engine = BacktestEngine({sym: df}, BACKTEST_CONFIG, use_regime_filter=False)
            engine.kelly_factors = {sym: 1.0}
            try:
                pf = engine._run_single(sym, df, kelly_mult=1.0)
                sharpe = _compute_sharpe(pf, BACKTEST_CONFIG["risk_free_rate"])
                ret = pf.total_return() * 100
                dd = pf.max_drawdown() * 100
            except Exception as e:
                sharpe, ret, dd = 0.0, 0.0, 0.0
            mb_rows.append({
                "Instrument": sym, "BreakoutPeriod": bp,
                "Sharpe": f"{sharpe:.2f}", "Return%": f"{ret:.2f}", "MaxDD%": f"{dd:.2f}",
            })
    if mb_rows:
        mdf = pd.DataFrame(mb_rows)
        print(mdf.to_string(index=False))
        mdf.to_csv(out_dir / "sweep_momentum_breakout.csv", index=False)
        print()

    # 3) Gold/Oil ratio: sweep std_threshold
    print("--- Gold/Oil Ratio: Sweeping std_threshold ---")
    go_rows = []
    if "GLD_USO_RATIO" in resampled:
        df = resampled["GLD_USO_RATIO"]
        std_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
        for std in std_vals:
            from config import STRATEGY_PARAMS as sp
            sp.setdefault("gold_oil_ratio", {}).setdefault("GLD_USO_RATIO", {})["std_threshold"] = std
            engine = BacktestEngine({"GLD_USO_RATIO": df}, BACKTEST_CONFIG, use_regime_filter=False)
            engine.kelly_factors = {"GLD_USO_RATIO": 1.0}
            try:
                pf = engine._run_single("GLD_USO_RATIO", df, kelly_mult=1.0)
                sharpe = _compute_sharpe(pf, BACKTEST_CONFIG["risk_free_rate"])
                ret = pf.total_return() * 100
                dd = pf.max_drawdown() * 100
            except Exception as e:
                sharpe, ret, dd = 0.0, 0.0, 0.0
            go_rows.append({
                "Instrument": "GLD_USO_RATIO", "StdThreshold": std,
                "Sharpe": f"{sharpe:.2f}", "Return%": f"{ret:.2f}", "MaxDD%": f"{dd:.2f}",
            })
    if go_rows:
        gdf = pd.DataFrame(go_rows)
        print(gdf.to_string(index=False))
        gdf.to_csv(out_dir / "sweep_gold_oil_std.csv", index=False)
        print()

    print("Sweep results saved to results/sweep_*.csv")
    print()

    # 4) Crypto momentum: sweep fast_ema
    print("--- Crypto Momentum: Sweeping fast_ema ---")
    cm_rows = []
    if "BTC/USD" in resampled:
        df = resampled["BTC/USD"]
        ema_vals = [10, 20, 30, 50]
        for fast in ema_vals:
            from config import STRATEGY_PARAMS as sp
            sp.setdefault("crypto_momentum", {}).setdefault("BTC/USD", {})["fast_ema"] = fast
            sp.setdefault("crypto_momentum", {}).setdefault("BTC/USD", {})["slow_ema"] = max(fast * 2.5, 50)
            engine = BacktestEngine({"BTC/USD": df}, BACKTEST_CONFIG, use_regime_filter=False)
            engine.kelly_factors = {"BTC/USD": 1.0}
            try:
                pf = engine._run_single("BTC/USD", df, kelly_mult=1.0)
                sharpe = _compute_sharpe(pf, BACKTEST_CONFIG["risk_free_rate"])
                ret = pf.total_return() * 100
                dd = pf.max_drawdown() * 100
            except Exception as e:
                sharpe, ret, dd = 0.0, 0.0, 0.0
            cm_rows.append({
                "Instrument": "BTC/USD", "FastEMA": fast,
                "Sharpe": f"{sharpe:.2f}", "Return%": f"{ret:.2f}", "MaxDD%": f"{dd:.2f}",
            })
    if cm_rows:
        cdf = pd.DataFrame(cm_rows)
        print(cdf.to_string(index=False))
        cdf.to_csv(out_dir / "sweep_crypto_ema.csv", index=False)
        print()


def _compute_trade_metrics(pf_portfolio) -> dict:
    """Compute robust trade-return-based metrics instead of broken daily-resample Sharpe.
    
    Returns:
        dict with trade_pf, trade_sharpe, trade_count, total_return
    """
    from reporting.metrics import bootstrap_sharpe
    trades = pf_portfolio.trades.count()
    total_ret = pf_portfolio.total_return()
    
    if trades < 3:
        return {"trade_pf": 0.0, "trade_sharpe": 0.0, "trade_count": trades, "total_return": total_ret}
    
    try:
        records = pf_portfolio.trades.records_readable
        ret_col = [c for c in records.columns if 'return' in c.lower()]
        if not ret_col:
            return {"trade_pf": float(total_ret), "trade_sharpe": 0.0, "trade_count": trades, "total_return": total_ret}
        
        rets = records[ret_col[0]].values.astype(float)
        
        # Profit factor from trades
        winners = rets[rets > 0].sum()
        losers = abs(rets[rets < 0].sum())
        trade_pf = winners / losers if losers > 1e-10 else float('inf')
        
        # Trade-return Sharpe with proper annualization
        avg_hold = 1.0
        if 'Entry Timestamp' in records.columns and 'Exit Timestamp' in records.columns:
            durations = (records['Exit Timestamp'] - records['Entry Timestamp']).dt.total_seconds() / (86400 * 5/7)
            avg_hold = max(float(durations.mean()), 0.5) if len(durations) > 0 else 1.0
        
        trade_sharpe = 0.0
        if len(rets) >= 5 and rets.std() > 1e-10:
            ann_factor = np.sqrt(252.0 / avg_hold)
            trade_sharpe = float(rets.mean() / rets.std() * ann_factor)
        
        return {"trade_pf": trade_pf, "trade_sharpe": trade_sharpe, 
                "trade_count": trades, "total_return": float(total_ret)}
    except Exception:
        return {"trade_pf": 0.0, "trade_sharpe": 0.0, "trade_count": trades, "total_return": float(total_ret)}


def _walk_forward_validate(symbol: str, df: pd.DataFrame) -> dict:
    """Per-instrument walk-forward: purged k-fold with embargo, PF-based scoring."""
    import pandas as pd

    n = len(df)
    n_splits = 10
    embargo = 20

    if n < 200:
        return {"symbol": symbol, "folds": 0, "recommendation": "INSUFFICIENT_DATA"}

    # Build purged train/test folds
    min_train = int(n * 0.4)
    test_size = (n - min_train) // (n_splits - 1)
    fold_boundaries = []
    train_end = min_train
    for i in range(n_splits - 1):
        test_start = train_end
        test_end = min(test_start + test_size, n)
        purge_end = min(test_start + embargo, test_end)
        fold_boundaries.append({
            "train": (0, train_end),
            "test_purged": (purge_end, test_end),
        })
        train_end = test_end
    if train_end < n:
        purge_end = min(train_end + embargo, n)
        fold_boundaries.append({
            "train": (0, train_end),
            "test_purged": (purge_end, n),
        })

    folds = []
    for i, fb in enumerate(fold_boundaries):
        is_slice = slice(fb["train"][0], fb["train"][1])
        oos_start, oos_end = fb["test_purged"]
        if oos_end - oos_start < 20:
            continue

        is_df = df.iloc[is_slice].copy()
        oos_df = df.iloc[oos_start:oos_end].copy()

        if len(is_df) < 60 or len(oos_df) < 15:
            continue

        try:
            engine_is = BacktestEngine({symbol: is_df}, BACKTEST_CONFIG, use_regime_filter=False)
            engine_is.kelly_factors = {symbol: 1.0}
            is_pf_obj = engine_is._run_single(symbol, is_df, kelly_mult=1.0)
            is_metrics = _compute_trade_metrics(is_pf_obj)
        except Exception:
            is_metrics = {"trade_pf": 0.0, "trade_sharpe": 0.0, "trade_count": 0, "total_return": 0.0}

        try:
            engine_oos = BacktestEngine({symbol: oos_df}, BACKTEST_CONFIG, use_regime_filter=False)
            engine_oos.kelly_factors = {symbol: 1.0}
            oos_pf_obj = engine_oos._run_single(symbol, oos_df, kelly_mult=1.0)
            oos_metrics = _compute_trade_metrics(oos_pf_obj)
        except Exception:
            oos_metrics = {"trade_pf": 0.0, "trade_sharpe": 0.0, "trade_count": 0, "total_return": 0.0}

        folds.append({
            "fold": i + 1,
            "is_bars": len(is_df),
            "oos_bars": len(oos_df),
            "is_trades": is_metrics["trade_count"],
            "oos_trades": oos_metrics["trade_count"],
            "is_pf": is_metrics["trade_pf"],
            "oos_pf": oos_metrics["trade_pf"],
            "pf_decay": is_metrics["trade_pf"] - oos_metrics["trade_pf"],
        })

    if not folds:
        return {"symbol": symbol, "folds": 0, "recommendation": "NO_FOLDS"}

    is_pfs = [f["is_pf"] for f in folds]
    oos_pfs = [f["oos_pf"] for f in folds]
    avg_is_pf = np.mean(is_pfs)
    avg_oos_pf = np.mean(oos_pfs)
    pf_decay = avg_is_pf - avg_oos_pf
    pos_windows = sum(1 for f in folds if f["oos_pf"] > 1.0)
    positive_ratio = pos_windows / max(len(folds), 1)

    # PF-based scoring
    if positive_ratio < 0.25:
        rec = "REJECT"
    elif pf_decay > 0.5 and avg_oos_pf < 1.0:
        rec = "OVERFIT"
    elif positive_ratio < 0.50:
        rec = "UNSTABLE"
    elif pf_decay > 0.5:
        rec = "DEGRADING"
    else:
        rec = "PASS"

    return {
        "symbol": symbol,
        "folds": len(folds),
        "avg_is_pf": round(avg_is_pf, 3),
        "avg_oos_pf": round(avg_oos_pf, 3),
        "pf_decay": round(pf_decay, 3),
        "positive_windows": pos_windows,
        "total_is_trades": sum(f["is_trades"] for f in folds),
        "total_oos_trades": sum(f["oos_trades"] for f in folds),
        "recommendation": rec,
        "fold_details": folds,
    }


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        parameter_sweep()
    elif "--challenge" in sys.argv:
        from prop_firm.challenge_lifecycle import ChallengeLifecycleManager
        from prop_firm.rule_simulator import simulate_challenge_sprint, print_sprint_report
        from config import CHALLENGE_CONFIG, SPRINT_STRATEGY_PARAMS, CHALLENGE_RISK_PARAMS

        print("=== PROP FIRM CHALLENGE SPRINT MODE ===")
        print(f"Account: ${CHALLENGE_CONFIG['initial_capital']:,}")
        print(f"Target: {CHALLENGE_CONFIG['profit_target_pct']:.0%} profit "
              f"(${CHALLENGE_CONFIG['profit_target_dollars']:,})")
        print(f"Max trading days: {CHALLENGE_CONFIG['max_trading_days']}")
        print(f"Daily DD limit: {CHALLENGE_CONFIG['daily_dd_limit_pct']:.1%} "
              f"(buffer: {CHALLENGE_CONFIG['max_dd_limit_pct']:.1%} total)")
        print(f"Instrument: {CHALLENGE_CONFIG['instrument']} ({CHALLENGE_CONFIG['timeframe']})")
        print()

        # Use existing data pipeline for SPY
        from data.fetcher import DataFetcher
        from data.resampler import resample_ohlcv
        from engine import BacktestEngine

        config = {**BACKTEST_CONFIG, "initial_capital": CHALLENGE_CONFIG["initial_capital"]}
        instruments = {"SPY": {"asset_class": "stock", "strategy": "prop_firm_sprint",
                               "base_tf": "1Min", "target_tf": CHALLENGE_CONFIG["timeframe"]}}

        # Temporarily register SPY with the engine's INSTRUMENTS lookup
        from config import INSTRUMENTS as _cfg_inst
        _cfg_inst["SPY"] = instruments["SPY"]

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        if api_key and api_key != "your_free_tier_key":
            fetcher = DataFetcher(api_key, secret_key)
            raw = fetcher.fetch_all(instruments, config)
        else:
            print("No Alpaca keys — generating synthetic SPY data for testing.")
            from data.synthetic import generate_synthetic_data
            raw = generate_synthetic_data(instruments, config)

        resampled = {
            sym: resample_ohlcv(df, info["target_tf"], info["asset_class"])
            for sym, info in instruments.items()
            for s, df in raw.items() if s == sym
        }

        if not resampled:
            print("ERROR: No data available for SPY sprint backtest.")
            sys.exit(1)

        engine = BacktestEngine(resampled, config, use_regime_filter=False)
        engine.kelly_factors = {"SPY": 1.0}
        portfolios = engine.run()

        if "SPY" in portfolios:
            result = simulate_challenge_sprint(
                portfolios["SPY"],
                initial_capital=CHALLENGE_CONFIG["initial_capital"],
                profit_target_pct=CHALLENGE_CONFIG["profit_target_pct"],
                daily_dd_limit_pct=CHALLENGE_CONFIG["daily_dd_limit_pct"],
                max_dd_limit_pct=CHALLENGE_CONFIG["max_dd_limit_pct"],
                max_trading_days=CHALLENGE_CONFIG["max_trading_days"],
            )
            print_sprint_report(result, label="SPY_Sprint")

            # Also run the lifecycle manager for phase simulation
            mgr = ChallengeLifecycleManager(
                initial_equity=CHALLENGE_CONFIG["initial_capital"],
                profit_target_pct=CHALLENGE_CONFIG["profit_target_pct"],
                daily_dd_limit_pct=CHALLENGE_CONFIG["daily_dd_limit_pct"],
                max_dd_limit_pct=CHALLENGE_CONFIG["max_dd_limit_pct"],
            )
            print("Phase Progression:")
            print(f"  {'Day':>4s} {'Equity':>10s} {'Phase':>14s} {'RiskMult':>8s} {'Trades':>6s} {'Status':>20s}")
            print(f"  {'-'*4} {'-'*10} {'-'*14} {'-'*8} {'-'*6} {'-'*20}")

            eq_series = portfolios["SPY"].value()
            if isinstance(eq_series, pd.DataFrame):
                eq_series = eq_series.iloc[:, 0]
            daily_eq = eq_series.resample("1D").last().dropna()

            for day_idx in range(1, min(len(daily_eq), 23)):
                eq = float(daily_eq.iloc[day_idx]) if day_idx < len(daily_eq) else 0
                phase_result = mgr.update(eq, day_idx)
                trades = phase_result.max_trades_today if phase_result.can_trade else 0
                print(f"  {day_idx:4d} ${eq:>8,.0f} {phase_result.phase:>14s} "
                      f"{phase_result.risk_multiplier:7.2f}x {trades:5d} "
                      f"{'TARGET HIT!' if phase_result.risk_multiplier==0 and phase_result.reason.startswith('TARGET') else phase_result.reason[:20]:>20s}")

    else:
        main()
