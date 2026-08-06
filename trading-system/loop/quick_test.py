import pandas as pd
import numpy as np
import itertools
import time

# Load data (smaller subset for faster testing)
train_df = pd.read_parquet('data/XAUUSD_1h_train.parquet')
print(f"Loaded {len(train_df)} rows")
df = train_df.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

# Reduce to 1/10th of data for quick test
df = df.iloc[::10]
print(f"Using {len(df)} rows for quick test")

# Parameter grid - expanded
param_grid = {
    'bb_period': [12, 14],
    'bb_std': [1.8, 2.0],
    'rsi_period': [9, 12],
    'atr_period': [14],
    'atr_regime_period': [50],
    'atr_regime_mult': [1.3, 1.5],
}

trend_period = 480

# Precompute all indicators once per parameter set
def compute_all_indicators(df, params):
    # Bollinger
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
    # Regime ATR
    df['atr_regime'] = tr.rolling(params['atr_regime_period']).mean()
    # Trend
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
        # Regime filter
        if row['atr'] > params['atr_regime_mult'] * row['atr_regime']:
            continue
        # Session filter: 6-21 UTC
        hour = row.name.hour
        if hour < 6 or hour > 21:
            continue
        if not in_pos:
            if row['close'] < row['bb_lower'] and row['rsi'] < 30 and row['close'] > row['trend']:
                in_pos = True; entry_type = 'long'; entry_price = row['close']; entry_time = row.name; continue
            if row['close'] > row['bb_upper'] and row['rsi'] > 70 and row['close'] < row['trend']:
                in_pos = True; entry_type = 'short'; entry_price = row['close']; entry_time = row.name; continue
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

results = []
keys = list(param_grid.keys())
values = list(param_grid.values())
total = len(list(itertools.product(*values)))
print(f"Testing {total} combinations...")

for i, v in enumerate(itertools.product(*values)):
    params = dict(zip(keys, v))
    df_tmp = df.copy()
    df_tmp = compute_all_indicators(df_tmp, params)
    pnls = generate_signals(df_tmp, params)
    if not pnls:
        continue
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins) / len(pnls)
    pf = abs(np.mean(wins) / np.mean(losses)) if np.mean(losses) != 0 else np.inf
    tr = sum(pnls)
    ex = np.mean(pnls)
    dd = np.max(np.maximum.accumulate(np.cumsum(pnls)) - np.cumsum(pnls))
    sh = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    
    pass_pro = (wr >= 0.70 and pf >= 1.5 and sh >= 2.0 and dd <= 0.10 and len(pnls) >= 100)
    status = "✅ PASS" if pass_pro else "❌"
    print(f"[{i+1}/{total}] {params} -> Trades:{len(pnls)} WR:{wr*100:.1f}% PF:{pf:.2f} Sh:{sh:.2f} DD:{dd*100:.1f}% {status}")
    results.append({**params, 'trades':len(pnls), 'wr':wr, 'pf':pf, 'sh':sh, 'dd':dd, 'ret':tr, 'ex':ex})

# Save
pd.DataFrame(results).to_csv('quick_test_results.csv', index=False)
print("Done. Results in quick_test_results.csv")