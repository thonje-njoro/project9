import pandas as pd
import numpy as np
import itertools
import time

# Load FULL data
train_df = pd.read_parquet('data/XAUUSD_1h_train.parquet')
val_df = pd.read_parquet('data/XAUUSD_1h_val.parquet')
print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

df = train_df.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

# Focus on most promising parameters from quick test
# bb_std=1.8, rsi_period=12 showed best PF (~1.45)
param_grid = {
    'bb_period': [12, 14],
    'bb_std': [1.8],
    'rsi_period': [12],
    'atr_period': [14],
    'atr_regime_period': [50],
    'atr_regime_mult': [1.3, 1.5, 1.8],  # Test more aggressive regime filter
}

trend_period = 480

def compute_all_indicators(df, params):
    bb_ma = df['close'].rolling(params['bb_period']).mean()
    bb_std = df['close'].rolling(params['bb_period']).std()
    df['bb_upper'] = bb_ma + bb_std * params['bb_std']
    df['bb_lower'] = bb_ma - bb_std * params['bb_std']
    df['bb_mid'] = bb_ma
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(params['rsi_period']).mean()
    avg_loss = loss.rolling(params['rsi_period']).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(params['atr_period']).mean()
    df['atr_regime'] = tr.rolling(params['atr_regime_period']).mean()
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
print(f"Testing {total} combinations on FULL train dataset...")
print("=" * 80)

for i, v in enumerate(itertools.product(*values)):
    params = dict(zip(keys, v))
    df_tmp = df.copy()
    df_tmp = compute_all_indicators(df_tmp, params)
    pnls = generate_signals(df_tmp, params)
    if not pnls:
        print(f"[{i+1}/{total}] {params} -> NO SIGNALS")
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
    
    pass_pro = (wr >= 0.70 and pf >= 1.5 and sh >= 2.0 and dd <= 0.10 and len(pnls) >= 100)
    status = "✅ PASS" if pass_pro else "❌"
    print(f"[{i+1}/{total}] {params}")
    print(f"  Trades: {len(pnls)} | WR: {wr*100:.1f}% | PF: {pf:.2f} | Sharpe: {sh:.2f} | MaxDD: {dd*100:.1f}% | Return: {tr*100:.2f}% | Expect: {ex*100:.3f}% | {status}")
    if not pass_pro:
        reasons = []
        if wr < 0.70: reasons.append(f"WR {wr*100:.1f}%")
        if pf < 1.5: reasons.append(f"PF {pf:.2f}")
        if sh < 2.0: reasons.append(f"Sh {sh:.2f}")
        if dd > 0.10: reasons.append(f"DD {dd*100:.1f}%")
        if len(pnls) < 100: reasons.append(f"Trades {len(pnls)}")
        print(f"  Fail: {', '.join(reasons)}")
    results.append({**params, 'trades':len(pnls), 'wr':wr, 'pf':pf, 'sh':sh, 'dd':dd, 'ret':tr, 'ex':ex})

# Save
pd.DataFrame(results).to_csv('full_train_test_results.csv', index=False)
print("\n" + "=" * 80)
print("FULL TRAIN TEST COMPLETE")
print("=" * 80)

# Also test on validation set for top 3 by trade count
print("\n\nVALIDATION SET TESTING FOR TOP CONFIGS...")
val_df_proc = val_df.copy()
val_df_proc['timestamp'] = pd.to_datetime(val_df_proc['timestamp'])
val_df_proc = val_df_proc.set_index('timestamp').sort_index()

# Sort by trades descending
top_configs = sorted(results, key=lambda x: x['trades'], reverse=True)[:3]
for i, r in enumerate(top_configs):
    params = {k: r[k] for k in keys}
    df_tmp = val_df_proc.copy()
    df_tmp = compute_all_indicators(df_tmp, params)
    pnls = generate_signals(df_tmp, params)
    if not pnls:
        print(f"Val [{i+1}] {params} -> NO SIGNALS")
        continue
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins) / len(pnls)
    pf = abs(np.mean(wins) / np.mean(losses)) if np.mean(losses) != 0 else np.inf
    tr = sum(pnls)
    ex = np.mean(pnls)
    cum = np.cumsum(pnls)
    dd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0
    sh = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    print(f"Val [{i+1}] {params}")
    print(f"  Trades: {len(pnls)} | WR: {wr*100:.1f}% | PF: {pf:.2f} | Sharpe: {sh:.2f} | MaxDD: {dd*100:.1f}% | Return: {tr*100:.2f}%")

print("\nResults saved to full_train_test_results.csv")