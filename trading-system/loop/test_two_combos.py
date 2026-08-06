import pandas as pd
import numpy as np

# Load data
train_df = pd.read_parquet('data/XAUUSD_1h_train.parquet')
val_df = pd.read_parquet('data/XAUUSD_1h_val.parquet')
print(f'Train: {len(train_df)} rows, Val: {len(val_df)} rows')

# Use train for testing
df = train_df.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Parameters to test
bb_period = 14
bb_std = 1.8
rsi_period = 12
atr_period = 14
atr_regime_period = 50
atr_regime_mult_list = [1.3, 1.5]
trend_period = 480
session_start = 6
session_end = 21

def test_atr_regime_mult(atr_regime_mult):
    # Compute indicators
    bb_ma = df['close'].rolling(bb_period).mean()
    bb_std_val = df['close'].rolling(bb_period).std()
    df['bb_upper'] = bb_ma + bb_std_val * bb_std
    df['bb_lower'] = bb_ma - bb_std_val * bb_std
    df['bb_mid'] = bb_ma
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()
    df['atr_regime'] = tr.rolling(atr_regime_period).mean()
    df['trend'] = df['close'].rolling(trend_period).mean()
    
    pnls = []
    in_pos = False
    entry_price = 0.0
    entry_type = None
    timestamps = df.index
    for i in range(1, len(df)):
        row = df.iloc[i]
        if pd.isna(row['bb_upper']) or pd.isna(row['rsi']) or pd.isna(row['atr']):
            continue
        if row['atr'] > atr_regime_mult * row['atr_regime']:
            continue
        hour = timestamps[i].hour
        if hour < session_start or hour > session_end:
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
                pnls.append(pnl)
                in_pos = False
    return pnls

print("Testing different atr_regime_mult values:")
for mult in atr_regime_mult_list:
    pnls = test_atr_regime_mult(mult)
    if not pnls:
        print(f"  mult={mult}: NO SIGNALS")
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
    print(f"  mult={mult}: Trades={len(pnls)} | WR={wr*100:.1f}% | PF={pf:.2f} | Sharpe={sh:.2f} | MaxDD={dd*100:.1f}% | Ret={tr*100:.2f}% | Expect={ex*100:.3f}%")