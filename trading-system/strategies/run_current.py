"""
Run the current candidate strategy on train / val / holdout splits.

Usage:
    python strategies/run_current.py            # default: train
    python strategies/run_current.py val
    python strategies/run_current.py holdout
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
from backtesting import Backtest
from strategies.candidate_current import DonchianBreakout  # <-- import the class only
from evaluator.daily_pnl import compute_daily_pnl_pct
from evaluator.prop_firm_rules import evaluate_against_rules

SPLITS = {
    "train":    "data/XAUUSD_15m_train.parquet",
    "val":      "data/XAUUSD_15m_val.parquet",
    "holdout":  "data/XAUUSD_15m_holdout.parquet",
}

INITIAL_CASH = 10_000
split_key = sys.argv[1] if len(sys.argv) > 1 else "train"

parquet = SPLITS.get(split_key)
if parquet is None:
    sys.exit(f"Unknown split '{split_key}'. Choose from: {list(SPLITS)}")

# --- load & prep data ---
df = pd.read_parquet(parquet)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
df = df.rename(columns={
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "volume": "Volume",
})

# --- run backtest ---
bt = Backtest(df, DonchianBreakout, cash=INITIAL_CASH, commission=0.0002,
              margin=1 / 30, finalize_trades=True)
stats = bt.run()

# --- display core stats ---
for key in ["Return [%]", "Max. Drawdown [%]", "# Trades", "Win Rate [%]",
            "Profit Factor", "Sharpe Ratio"]:
    print(f"{key:30s} {stats[key]}")

# --- prop firm evaluation ---
daily_pnl = compute_daily_pnl_pct(stats["_equity_curve"], INITIAL_CASH)
print()
for firm in ["ftmo_2step_phase1", "the5ers_high_stakes", "fundingpips_2step_phase1"]:
    res = evaluate_against_rules(stats, daily_pnl, firm)
    tag = "PASS" if res["passed"] else "FAIL"
    print(f"[{tag}] {firm:35s}  ret={res['return_pct']:.2f}%  dd={res['max_drawdown_pct']:.2f}%")
    for f in res["failures"]:
        print(f"       {f}")
