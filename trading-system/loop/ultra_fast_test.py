import pandas as pd
import numpy as np
import itertools

# Load data - just first 2000 rows for ultra-fast test
train_df = pd.read_parquet('data/XAUUSD_1h_train.parquet')
df = train_df.iloc[:2000].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()
print(f"Using {len(df)} rows for ultra-fast test (no regime filter)")

# Parameter grid - reduced
param_grid = {
    'bb_period': [10, 12, 14, 16, 20],
    'bb_std': [1.8, 2.0, 2.2, 2.5],
    'rsi_period': [10, 12, 14],
}

# Fixed parameters
atr_period = 14
trend_period = 480
session_start_hour = 6
session_end_hour = 21

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
    df['atr'] = tr.rolling(atr_period).mean()
    # Trend (SMA)
    df['trend'] = df['close'].rolling(trend_period).mean()
    return df

def generate_signals(df, params):
    signals = []
    in_pos = False
    entry_price = 0.0
    entry_type = None
    for i in range(1, len(df)):
        row = df.iloc[i]
        if pd.isna(row['bb_upper']) or pd.isna(row['rsi']) or pd.isna(row['atr']):
            continue
        hour = df.index[i].hour
        if hour < session_start_hour or hour > session_end_hour:
            continue
        if not in_pos:
            if row['close'] < row['bb_lower'] and row['rsi'] < 30 and row['close'] > row['trend']:
                in_pos = True; entry_type = 'long'; entry_price = row['close']; continue
            if row['close'] > row['bb_upper'] and row['rsi'] > 70 and row['close'] < row['trend']:
                in_pos = True; entry_type = 'short'; entry_price = row['close']; continue
        if in_pos:
            exit_sig = False
            if entry_type == 'long':
                if row['close'] >= row['bb_mid'] or row['rsi'] > 70: exit_sig = True
            else:
                if row['close'] <= row['bb_mid'] or row['rsi'] < 30: exit_sig = True
            if exit_sig:
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price if entry_type == 'long' else (entry_price - exit_price) / entry_price
                signals.append(pnl)
                in_pos = False
    return signals

keys, values = zip(*param_grid.items())
total = len(list(itertools.product(*values)))
print(f"Testing {total} combinations...")

for i, v in enumerate(itertools.product(*values)):
    params = dict(zip(keys, v))
    df_tmp = df.copy()
    df_tmp = compute_indicators(df_tmp, params)
    pnls = generate_signals(df_tmp, params)
    if not pnls:
        continue
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins) / len(pnls)
    pf = abs(np.mean(wins) / np.mean(losses)) if np.mean(losses) != 0 else np.inf
    tr = sum(pnls)
    ex = np.mean(pnls)
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = np.max(running_max - cum) if len(cum) > 0 else 0
    sh = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    results.append({
        **params,
        'trades': len(pnls),
        'win_rate': wr,
        'profit_factor': pf,
        'expectancy': ex,
        'total_return': tr,
        'max_dd': dd,
        'sharpe': sh,
    })

results_sorted = sorted(results, key=lambda x: (x['win_rate'], x['profit_factor']), reverse=True)
print("\nTop 10 configurations by win rate:")
for i, r in enumerate(results_sorted[:10]):
    print(f"{i+1}. BB{r['bb_period']}, STD{r['bb_std']}, RSI{r['rsi_period']} -> "
          f"Trades: {r['trades']}, WR: {r['win_rate']*100:.1f}%, PF: {r['profit_factor']:.2f}, "
          f"Sharpe: {r['sharpe']:.2f}, MaxDD: {r['max_dd']*100:.1f}%, Return: {r['total_return']*100:.2f}%")

prof_passing = [r for r in results if (
    r['win_rate'] >= 0.70 and
    r['profit_factor'] >= 1.5 and
    r['sharpe'] >= 2.0 and
    r['max_dd'] <= 0.10 and
    r['trades'] >= 100
)]
print(f"\nNumber of configurations meeting professional criteria: {len(prof_passing)}")

pd.DataFrame(results).to_csv('no_regime_results.csv', index=False)
print("\nResults saved to no_regime_results.csv")