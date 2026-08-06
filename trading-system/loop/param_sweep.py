#!/usr/bin/env python3
"""
Efficient Parameter Sweep for Trading Strategy
Fixed dynamic strategy creation.
"""

import sys
import os
import json
import itertools
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
from backtesting import Backtest, Strategy

from evaluator.daily_pnl import compute_daily_pnl_pct
from evaluator.prop_firm_rules import evaluate_against_rules
from loop.scorer import score_candidate

INITIAL_CASH = 10_000
FIRMS = ["ftmo_2step_phase1", "the5ers_high_stakes", "fundingpips_2step_phase1"]

# Parameter sweep grid (reduced for speed)
PARAMETER_SWEEP = {
    "bb_period":         [14, 16],
    "bb_std":            [2.0, 2.2],
    "rsi_period":        [10, 14],
}

# Fixed parameters (not swept)
FIXED_PARAMS = {
    "atr_period": 14,
    "trend_period": 480,
    "risk_pct": 0.015,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 4.0,
    "atr_regime_period": 50,
    "atr_regime_mult": 1.5,
}

SYMBOL_PREFIX = "XAUUSD_1h"
RESULTS_DIR = Path("runs/param_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def make_strategy(params):
    """Create a strategy class dynamically using type()"""
    
    def init(self):
        close = pd.Series(self.data.Close)
        high  = pd.Series(self.data.High)
        low   = pd.Series(self.data.Low)

        self._close = close
        self._high  = high
        self._low   = low

        self.bb_mid  = self.I(lambda: close.rolling(self.bb_period).mean(), name="BB_MID")
        self.bb_std  = self.I(lambda: close.rolling(self.bb_period).std(),  name="BB_STD")
        self.rsi     = self.I(self._rsi, name="RSI")
        self.atr     = self.I(self._atr, name="ATR")
        self.trend   = self.I(lambda: close.rolling(self.trend_period).mean(), name="TREND")
        # Regime detection ATR (50-period)
        self._atr_regime = self.I(
            lambda: pd.concat([
                self._high - self._low,
                (self._high - self._close.shift()).abs(),
                (self._low - self._close.shift()).abs(),
            ], axis=1).max(axis=1).rolling(self.atr_regime_period).mean(),
            name="ATR_REGIME"
        )

    def _rsi(self):
        delta = self._close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta).clip(lower=0).rolling(self.rsi_period).mean()
        return (100 - 100 / (1 + gain / loss)).values

    def _atr(self):
        tr = pd.concat([
            self._high - self._low,
            (self._high - self._close.shift()).abs(),
            (self._low - self._close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean().values

    def next(self):
        close = self.data.Close[-1]
        bb_mid = self.bb_mid[-1]
        bb_std = self.bb_std[-1]
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        trend = self.trend[-1]

        # Regime filter: skip mean-reversion entries in trending/volatile markets
        atr_short = self.atr[-1]
        atr_long = self._atr_regime[-1]
        if atr_short > atr_long * self.atr_regime_mult:
            return  # trending regime — skip entry

        # Session filter: only trade London/NY overlap (7 AM to 8 PM UTC)
        hour = self.data.index[-1].hour
        if not (7 <= hour <= 20):
            return

        sl_distance = atr * self.sl_atr_mult
        size_fraction = (self.risk_pct * close * (1/30)) / sl_distance
        size_fraction = max(0.001, min(0.10, size_fraction))

        if close < bb_lower and rsi < 30 and close > trend:
            tp_distance = sl_distance * (self.tp_atr_mult / self.sl_atr_mult)
            self.buy(size=size_fraction, sl=close - sl_distance, tp=close + tp_distance)
        elif close > bb_upper and rsi > 70 and close < trend:
            tp_distance = sl_distance * (self.tp_atr_mult / self.sl_atr_mult)
            self.sell(size=size_fraction, sl=close + sl_distance, tp=close - tp_distance)

    # Create class dynamically
    cls_name = f"SweepStrategy_{params['bb_period']}_{params['bb_std']}_{params['rsi_period']}"
    cls_dict = {
        'bb_period': params['bb_period'],
        'bb_std': params['bb_std'],
        'rsi_period': params['rsi_period'],
        'atr_period': params['atr_period'],
        'trend_period': params['trend_period'],
        'risk_pct': params['risk_pct'],
        'sl_atr_mult': params['sl_atr_mult'],
        'tp_atr_mult': params['tp_atr_mult'],
        'atr_regime_period': params['atr_regime_period'],
        'atr_regime_mult': params['atr_regime_mult'],
        'init': init,
        '_rsi': _rsi,
        '_atr': _atr,
        'next': next,
    }
    
    return type(cls_name, (Strategy,), cls_dict)


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
    win_rate = val_stats.get('win_rate', 0) * 100
    profit_factor = val_stats.get('profit_factor', 0)
    sharpe = val_stats.get('sharpe_ratio', 0)
    max_dd = -val_stats.get('max_relative_drawdown', 0) * 100
    trades = val_stats.get('trades', 0)
    total_return = val_stats.get('total_return', 0) * 100
    calmar = total_return / max_dd if max_dd != 0 else 0

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
        return {'keep': False, 'reasons': reasons, 'metrics': {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'calmar': calmar,
            'max_dd': max_dd,
            'trades': trades,
            'total_return': total_return
        }}

def run_backtest_and_eval(strategy_class, df):
    bt = Backtest(df, strategy_class, cash=INITIAL_CASH, commission=0.0002,
                  margin=1/30, finalize_trades=True)
    stats = bt.run()
    daily_pnl = compute_daily_pnl_pct(stats["_equity_curve"], INITIAL_CASH)
    evals = [evaluate_against_rules(stats, daily_pnl, firm) for firm in FIRMS]
    return stats, evals

def run_sweep():
    train_df = load_split(SYMBOL_PREFIX, "train")
    val_df = load_split(SYMBOL_PREFIX, "val")
    
    print(f"Starting sweep with {len(list(itertools.product(*PARAMETER_SWEEP.values())))} combinations")
    print("=" * 60)
    
    results = []
    
    for i, combo in enumerate(itertools.product(*PARAMETER_SWEEP.values())):
        params = {k: v for k, v in zip(PARAMETER_SWEEP.keys(), combo)}
        all_params = {**FIXED_PARAMS, **params}
        
        print(f"[{i+1}/{len(list(itertools.product(*PARAMETER_SWEEP.values())))}] Testing: {params}")
        
        try:
            strategy_class = make_strategy(all_params)
            train_stats, train_evals = run_backtest_and_eval(strategy_class, train_df)
            val_stats, val_evals = run_backtest_and_eval(strategy_class, val_df)
            result = score_candidate(train_stats, train_evals, val_stats, val_evals)
            prof_validation = validate_professional_metrics(val_stats)
            
            output = {
                "iteration": i+1,
                "params": all_params,
                "sweep_params": params,
                "train_return_pct": result.get("train_return_pct", 0),
                "val_return_pct": result.get("val_return_pct", 0),
                "professional_keep": prof_validation['keep'],
                "professional_reasons": prof_validation['reasons'],
                "metrics": prof_validation['metrics'],
                "train_evals": train_evals,
                "val_evals": val_evals
            }
            results.append(output)
            
            if (i+1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(list(itertools.product(*PARAMETER_SWEEP.values())))}")
                
        except Exception as e:
            results.append({
                "params": all_params,
                "error": str(e),
                "professional_keep": False
            })
            print(f"  ERROR: {e}")
    
    # Save results
    results_file = RESULTS_DIR / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    kept = [r for r in results if r.get('professional_keep', False)]
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Total combinations: {len(results)}")
    print(f"Professional passes: {len(kept)}")
    if kept:
        best = max(kept, key=lambda x: x['val_return_pct'])
        print(f"Best return: {best['val_return_pct']:.2f}% with {best['sweep_params']}")
    
    print(f"\nFull results: {results_file}")
    return results

if __name__ == "__main__":
    run_sweep()