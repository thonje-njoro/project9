import sys, os, json, importlib, shutil, math
sys.path.insert(0, ".")
import pandas as pd
from backtesting import Backtest, Strategy

from evaluator.daily_pnl import compute_daily_pnl_pct
from evaluator.prop_firm_rules import evaluate_against_rules, PROP_FIRM_RULES
from loop.scorer import score_candidate

INITIAL_CASH = 10_000
FIRMS = ["ftmo_2step_phase1", "the5ers_high_stakes", "fundingpips_2step_phase1"]

def load_split(symbol_prefix, split_name):
    path = f"data/{symbol_prefix}_{split_name}.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })
    return df

def validate_professional_metrics(val_stats):
    # Extract relevant metrics
    win_rate = val_stats.get('win_rate', 0) * 100
    profit_factor = val_stats.get('profit_factor', 0)
    sharpe = val_stats.get('sharpe_ratio', 0)
    max_dd = -val_stats.get('max_relative_drawdown', 0) * 100  # convert to percent
    trades = val_stats.get('trades', 0)
    total_return = val_stats.get('total_return', 0) * 100
    # Calculate Calmar ratio
    calmar = total_return / max_dd if max_dd != 0 else 0
    # Professional metric thresholds
    thresholds = {
        'win_rate': 70,
        'profit_factor': 1.5,
        'sharpe': 2.0,
        'calmar': 1.0,
        'max_dd': 10,
        'trades': 100
    }
    reasons = []
    if win_rate < thresholds['win_rate']:
        reasons.append('Win rate below 70%')
    if profit_factor < thresholds['profit_factor']:
        reasons.append('Profit factor below 1.5')
    if sharpe < thresholds['sharpe']:
        reasons.append('Sharpe below 2.0')
    if calmar < thresholds['calmar']:
        reasons.append('Calmar below 1.0')
    if max_dd > thresholds['max_dd']:
        reasons.append('Max drawdown above 10%')
    if trades < thresholds['trades']:
        reasons.append('Insufficient trade count')
    if not reasons:
        return {'keep': True, 'reasons': []}
    else:
        return {'keep': False, 'reasons': reasons}

def run_backtest_and_eval(strategy_class, df):
    bt = Backtest(df, strategy_class, cash=INITIAL_CASH, commission=0.0002,
                   margin=1/30, finalize_trades=True)
    stats = bt.run()
    daily_pnl = compute_daily_pnl_pct(stats["_equity_curve"], INITIAL_CASH)
    evals = [evaluate_against_rules(stats, daily_pnl, firm) for firm in FIRMS]
    return stats, evals

def find_strategy_class(mod):
    """Find the single Strategy subclass in a module."""
    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if (isinstance(obj, type) and issubclass(obj, Strategy) and
                obj is not Strategy):
            candidates.append(obj)

    # Ensure exactly one Strategy class found
    if not candidates:
        raise ValueError(f"No Strategy subclass found in {mod.__name__}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple Strategy subclasses found in {mod.__name__}: {[c.__name__ for c in candidates]}"
        )

    return candidates[0]

def run_iteration(iteration_num, strategy_module_path, symbol_prefix="EURUSD_15m"):
    run_dir = f"runs/iteration_{iteration_num:04d}"
    os.makedirs(run_dir, exist_ok=True)
    src_path = strategy_module_path.replace(".", "/") + ".py"
    shutil.copy(src_path, f"{run_dir}/strategy.py")
    mod = importlib.import_module(strategy_module_path)
    importlib.reload(mod)
    strategy_class = find_strategy_class(mod)
    train_df = load_split(symbol_prefix, "train")
    val_df = load_split(symbol_prefix, "val")
    train_stats, train_evals = run_backtest_and_eval(strategy_class, train_df)
    val_stats, val_evals = run_backtest_and_eval(strategy_class, val_df)
    result = score_candidate(train_stats, train_evals, val_stats, val_evals)
    # Apply professional metric validation
    professional_validation = validate_professional_metrics(val_stats)
    if not professional_validation['keep']:
        result['keep'] = False
        if 'reasons' not in result or not result['reasons']:
            result['reasons'] = []
        result['reasons'].extend(professional_validation['reasons'])
    output = {
        "iteration": iteration_num,
        "keep": result["keep"],
        "reasons": result.get("reasons", []),
        "train_return_pct": result.get("train_return_pct", 0),
        "val_return_pct": result.get("val_return_pct", 0),
        "robustness_gap": result.get("robustness_gap", 0),
        "score": result.get("score", 0),
        "train_evals": train_evals,
        "val_evals": val_evals,
    }
    with open(f"{run_dir}/result.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Iteration {iteration_num}: keep={result['keep']}  train={result['train_return_pct']:.2f}%  val={result['val_return_pct']:.2f}%")
    if result["reasons"]:
        for r in result["reasons"]:
            print(f"  - {r}")
    return output

if __name__ == "__main__":
    run_iteration(1, "strategies.baseline_sma_cross", symbol_prefix="XAUUSD_15m")