import pandas as pd
import numpy as np
import itertools

# Load data
train_df = pd.read_parquet('data/XAUUSD_1h_train.parquet')
val_df = pd.read_parquet('data/XAUUSD_1h_val.parquet')
df = train_df.copy()

# Ensure datetime index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

# Parameter grid - expanded based on action items
param_grid = {
    'bb_period': [12, 14],
    'bb_std': [1.8, 2.0],
    'rsi_period': [9, 12],
    'atr_period': [14],
    'atr_regime_period': [50],
    'atr_regime_mult': [1.3, 1.5],
    'risk_pct': [0.015],
    'sl_atr_mult': [2.0],
    'tp_atr_mult': [4.0],
}

# Fixed parameters
trend_period = 480

results = []

def compute_indicators(df, params):
    # Bollinger Bands
    bb_ma = df['close'].rolling(params['bb_period']).mean()
    bb_std = df['close'].rolling(params['bb_period']).std()
    df['bb_upper'] = bb_ma + bb_std * params['bb_std']
    df['bb_lower'] = bb_ma - bb_std * params['bb_std']
    df['bb_mid'] = bb_ma
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(params['rsi_period']).mean()
    avg_loss = loss.rolling(params['rsi_period']).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(params['atr_period']).mean()
    # Regime ATR (longer)
    df['atr_regime'] = tr.rolling(params['atr_regime_period']).mean()
    # Trend (SMA)
    df['trend'] = df['close'].rolling(trend_period).mean()
    return df

def generate_signals(df, params):
    signals = []
    in_position = False
    entry_price = 0.0
    entry_type = None
    for i in range(1, len(df)):
        row = df.iloc[i]
        # Skip if any indicator is NaN
        if pd.isna(row['bb_upper']) or pd.isna(row['rsi']) or pd.isna(row['atr']):
            continue
        # Regime filter
        if row['atr'] > params['atr_regime_mult'] * row['atr_regime']:
            continue
        # Session filter: 6 AM to 21 PM UTC (expanded)
        hour = row.name.hour
        if hour < 6 or hour > 21:
            continue
        # Entry conditions
        if not in_position:
            # Long
            if row['close'] < row['bb_lower'] and row['rsi'] < 30 and row['close'] > row['trend']:
                in_position = True
                entry_type = 'long'
                entry_price = row['close']
                entry_time = row.name
                continue
            # Short
            if row['close'] > row['bb_upper'] and row['rsi'] > 70 and row['close'] < row['trend']:
                in_position = True
                entry_type = 'short'
                entry_price = row['close']
                entry_time = row.name
                continue
        # Exit conditions
        if in_position:
            exit_signal = False
            if entry_type == 'long':
                if row['close'] >= row['bb_mid'] or row['rsi'] > 70:
                    exit_signal = True
            else:
                if row['close'] <= row['bb_mid'] or row['rsi'] < 30:
                    exit_signal = True
            if exit_signal:
                exit_price = row['close']
                if entry_type == 'long':
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                signals.append({
                    'entry_time': entry_time,
                    'exit_time': row.name,
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                })
                in_position = False
    return signals

# Iterate over parameter combinations
keys, values = zip(*param_grid.items())
total_combos = len(list(itertools.product(*values)))
print(f"Total combinations to test: {total_combos}")
print("=" * 60)

for i, v in enumerate(itertools.product(*values)):
    params = dict(zip(keys, v))
    df_tmp = df.copy()
    df_tmp = compute_indicators(df_tmp, params)
    signals = generate_signals(df_tmp, params)
    if not signals:
        print(f"[{i+1}/{total_combos}] {params} -> NO SIGNALS")
        continue
    pnls = [s['pnl'] for s in signals]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    total_return = sum(pnls)
    expectancy = np.mean(pnls) if pnls else 0
    # Max drawdown
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    drawdown = (running_max - cum)
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
    # Sharpe ratio (daily returns approximation)
    if len(pnls) > 1:
        daily_returns = np.array(pnls)
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    else:
        sharpe = 0
    
    result = {
        'params': params,
        'trades': len(signals),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'total_return': total_return,
        'max_dd': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
    }
    results.append(result)
    
    # Professional metric check
    professional_pass = (
        win_rate >= 0.70 and
        profit_factor >= 1.5 and
        sharpe >= 2.0 and
        max_dd <= 0.10 and
        len(signals) >= 100
    )
    
    status = "✅ PROFESSIONAL PASS" if professional_pass else "❌ FAIL"
    print(f"[{i+1}/{total_combos}] {params}")
    print(f"  Trades: {len(signals)} | Win Rate: {win_rate*100:.1f}% | PF: {profit_factor:.2f} | Sharpe: {sharpe:.2f} | Max DD: {max_dd*100:.1f}% | Return: {total_return*100:.2f}% | {status}")
    if not professional_pass:
        reasons = []
        if win_rate < 0.70: reasons.append(f"WR {win_rate*100:.1f}% < 70%")
        if profit_factor < 1.5: reasons.append(f"PF {profit_factor:.2f} < 1.5")
        if sharpe < 2.0: reasons.append(f"Sharpe {sharpe:.2f} < 2.0")
        if max_dd > 0.10: reasons.append(f"Max DD {max_dd*100:.1f}% > 10%")
        if len(signals) < 100: reasons.append(f"Trades {len(signals)} < 100")
        print(f"  Reasons: {', '.join(reasons)}")

# Sort by professional pass first, then by win_rate, then profit_factor
results_sorted = sorted(results, key=lambda x: (
    -(x['win_rate'] >= 0.70 and x['profit_factor'] >= 1.5 and x['sharpe'] >= 2.0 and x['max_dd'] <= 0.10 and x['trades'] >= 100),
    -x['win_rate'], -x['profit_factor'], -x['sharpe']
))

print("\n" + "=" * 60)
print("FINAL RESULTS - TOP CANDIDATES")
print("=" * 60)

for i, r in enumerate(results_sorted[:5]):
    print(f"\n{i+1}. Params: {r['params']}")
    print(f"   Trades: {r['trades']}")
    print(f"   Win Rate: {r['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {r['profit_factor']:.2f}")
    print(f"   Sharpe: {r['sharpe']:.2f}")
    print(f"   Max Drawdown: {r['max_dd']*100:.2f}%")
    print(f"   Total Return: {r['total_return']*100:.2f}%")
    print(f"   Expectancy: {r['expectancy']:.4f}")

# Save results
results_df = pd.DataFrame([
    {
        **r['params'],
        'trades': r['trades'],
        'win_rate': r['win_rate'],
        'profit_factor': r['profit_factor'],
        'expectancy': r['expectancy'],
        'total_return': r['total_return'],
        'max_dd': r['max_dd'],
        'avg_win': r['avg_win'],
        'avg_loss': r['avg_loss'],
        'sharpe': r['sharpe'],
    }
    for r in results
])
results_df.to_csv('param_sweep_results_expanded.csv', index=False)
print(f"\nResults saved to param_sweep_results_expanded.csv")

# Save professional passing candidates
prof_passing = [r for r in results if (
    r['win_rate'] >= 0.70 and
    r['profit_factor'] >= 1.5 and
    r['sharpe'] >= 2.0 and
    r['max_dd'] <= 0.10 and
    r['trades'] >= 100
)]
if prof_passing:
    prof_df = pd.DataFrame(prof_passing)
    prof_df.to_csv('professional_passing_params.csv', index=False)
    print(f"Professional passing candidates saved to professional_passing_params.csv ({len(prof_passing)} configs)")
else:
    print("⚠️ No candidates passed all professional thresholds")