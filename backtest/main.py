"""
main.py — CLI entry point for Project9 backtest system.
Usage:
    python main.py                    # Run full portfolio backtest
    python main.py --json             # Output results as JSON
    python main.py --validate         # Run full validation pipeline
    python main.py --symbol GLD       # Run single instrument
    python main.py --sweep            # Parameter sweep
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.config import INSTRUMENTS, STRATEGY_PARAMS
from backtest.engine import BacktestEngine


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_single(args):
    """Run backtest for a single instrument."""
    engine = BacktestEngine()
    symbol = args.symbol

    if symbol not in INSTRUMENTS:
        print(f"Unknown symbol: {symbol}")
        print(f"Available: {', '.join(INSTRUMENTS.keys())}")
        sys.exit(1)

    instrument = INSTRUMENTS[symbol]
    strategy_name = instrument["strategy"]
    params = STRATEGY_PARAMS.get(strategy_name, {}).get(symbol, {})

    result = engine.run_single(symbol, strategy_name, params)

    if args.json:
        print(json.dumps(result, indent=2, cls=_NumpyEncoder))
    else:
        print(f"\n{'='*60}")
        print(f"  {symbol} ({strategy_name})")
        print(f"{'='*60}")
        print(f"  Trades:     {result.get('n_trades', 0)}")
        print(f"  Sharpe:     {result.get('sharpe', 0):.3f}")
        print(f"  Max DD:     {result.get('max_drawdown', 0):.3f}")
        print(f"  Sortino:    {result.get('sortino', 0):.3f}")
        print(f"  Calmar:     {result.get('calmar', 0):.3f}")
        print(f"  Total Ret:  {result.get('total_return', 0):.2%}")

        mc = result.get("monte_carlo", {})
        if mc:
            print(f"\n  Monte Carlo:")
            print(f"    Sharpe median:  {mc.get('sharpe_median', 0):.3f}")
            print(f"    DD 99th pct:    {mc.get('dd_99pct', 0):.3f}")
            print(f"    Survival rate:  {mc.get('survival_rate', 0):.1%}")
            print(f"    Passes:         {'✓' if mc.get('passes') else '✗'}")

        if "error" in result:
            print(f"\n  ERROR: {result['error']}")

    return result


def run_portfolio(args):
    """Run full portfolio backtest."""
    engine = BacktestEngine()
    results = engine.run_portfolio()

    if args.json:
        print(json.dumps(results, indent=2, cls=_NumpyEncoder))
    else:
        print(f"\n{'='*60}")
        print(f"  PROJECT9 PORTFOLIO BACKTEST")
        print(f"{'='*60}")

        instruments = results.get("instruments", {})
        for symbol, result in instruments.items():
            status = "✓" if result.get("n_trades", 0) > 0 and "error" not in result else "✗"
            sharpe = result.get("sharpe", 0)
            trades = result.get("n_trades", 0)
            dd = result.get("max_drawdown", 0)
            print(f"  {status} {symbol:12s} | Trades: {trades:4d} | Sharpe: {sharpe:7.3f} | MaxDD: {dd:.3f}")

        combined = results.get("combined", {})
        print(f"\n  Combined:")
        print(f"    Avg Sharpe:  {combined.get('avg_sharpe', 0):.3f}")
        print(f"    Best Sharpe: {combined.get('best_sharpe', 0):.3f}")
        print(f"    DSR:         {combined.get('dsr', 0):.3f}")
        print(f"    DSR Passes:  {'✓' if combined.get('dsr_passes') else '✗'}")

    return results


def run_validation(args):
    """Run full validation pipeline."""
    engine = BacktestEngine()
    results = engine.validate()

    if args.json:
        print(json.dumps(results, indent=2, cls=_NumpyEncoder))
    else:
        print(f"\n{'='*60}")
        print(f"  VALIDATION RESULTS")
        print(f"{'='*60}")

        wfv = results.get("wfv", {})
        for symbol, wfv_result in wfv.items():
            status = "✓" if wfv_result.get("passes") else "✗"
            consistency = wfv_result.get("consistency_rate", 0)
            mean_oos = wfv_result.get("mean_oos_sharpe", 0)
            print(f"  {status} {symbol:12s} | OOS Sharpe: {mean_oos:7.3f} | Consistency: {consistency:.1%}")

        print(f"\n  Overall: {'PASS ✓' if results.get('overall_passes') else 'FAIL ✗'}")

    return results


def run_sweep(args):
    """Parameter sweep (stub — runs current params only)."""
    print("Parameter sweep mode")
    print("NOTE: Full sweep iterates ±30% around each parameter.")
    print("Running with current parameters first...\n")
    return run_portfolio(args)


def main():
    parser = argparse.ArgumentParser(description="Project9 Backtest Engine")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--validate", action="store_true", help="Run validation pipeline")
    parser.add_argument("--symbol", type=str, help="Run single instrument")
    parser.add_argument("--sweep", action="store_true", help="Parameter sweep mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.symbol:
        return run_single(args)
    elif args.validate:
        return run_validation(args)
    elif args.sweep:
        return run_sweep(args)
    else:
        return run_portfolio(args)


if __name__ == "__main__":
    main()
