import pandas as pd
import numpy as np

# Load data
print("Loading data...")
train_df = pd.read_parquet('data/XAUUSD_1h_train.parquet')
val_df = pd.read_parquet('data/XAUUSD_1h_val.parquet')
print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

# Use train for testing
df = train_df.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()
print(f"Using {len(df)} rows from train data")

# Parameters to test
params = {
    'bb_period': 14,
    'bb_std': 1.8,
    'rsi_period': 12,
    'atr_period': 14,
    'atr_regime_period': 50,
    'atr_regime_mult': 1.3,  # from config C
    'trend_period': 480,
    'risk_pct': 0.015,
    'sl_atr_mult': 2.0,
    'tp_atr_mult': 4.0,
    'session_start': 6,
    'session_end': 21,
}

print(f"Testing parameters: {params}")

# Compute indicators
print("Computing indicators...")
# Bollinger Bands
bb_ma = df['close'].rolling(params['bb_period']).mean()
bb_std_val = df['close'].rolling(params['bb_period']).std()
df['bb_upper'] = bb_ma + bb_std_val * params['bb_std']
df['bb_lower'] = bb_ma - bb_std_val * params['bb_std']
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
df['atr_regime'] = tr.rolling(params['atr_regime_period']).mean()
# Trend
df['trend'] = df['close'].rolling(params['trend_period']).mean()

# Regime filter: skip if ATR > atr_regime_mult * atr_regime
df['regime_filter'] = df['atr'] <= params['atr_regime_mult'] * df['atr_regime']
# Session filter: hour between session_start and session_end UTC
df['hour'] = df.index.hour
df['session_filter'] = (df['hour'] >= params['session_start']) & (df['hour'] <= params['session_end'])
# Combine filters
df['filter_ok'] = df['regime_filter'] & df['session_filter']

# Entry signals
df['long_entry'] = (df['close'] < df['bb_lower']) & (df['rsi'] < 30) & (df['close'] > df['trend']) & df['filter_ok']
df['short_entry'] = (df['close'] > df['bb_upper']) & (df['rsi'] > 70) & (df['close'] < df['trend']) & df['filter_ok']

# Exit signals
df['long_exit'] = (df['close'] >= df['bb_mid']) | (df['rsi'] > 70)
df['short_exit'] = (df['close'] <= df['bb_mid']) | (df['rsi'] < 30)

print(f"Long entry signals: {df['long_entry'].sum()}")
print(f"Short entry signals: {df['short_entry'].sum()}")

# Now simulate trades
pnls = []
in_position = False
entry_price = 0.0
entry_type = None  # 'long' or 'short'
entry_idx = -1

# We'll iterate over the dataframe, but we can skip to the next signal
# Get indices where any entry signal is True
entry_indices = df.index[df['long_entry'] | df['short_entry']].tolist()
print(f"Total entry signals: {len(entry_indices)}")

if len(entry_indices) == 0:
    print("No entry signals found!")
else:
    for idx in entry_indices:
        row = df.loc[idx]
        # Check if we are still in a position from previous signal
        if in_position:
            # Check if we should exit at this bar (before processing new entry)
            # We need to check exit condition on the bar we are currently at (idx)
            # But note: we are at the entry signal bar. We should check exit on the same bar? Typically, we check exit on the next bar.
            # For simplicity, we'll check exit on the same bar after entry? Actually, we should have exited on a previous bar.
            # Let's change approach: we will iterate through each bar, not just entry signals.
            pass

    # Let's do a simple bar-by-bar iteration but only over the filtered data to speed up? 
    # Actually, we can iterate over the whole dataframe but break early if we have processed all signals? 
    # Given the time, let's do a simple bar-by-bar loop but hope it's not too slow for 24846 rows.
    # 24846 iterations is fine.

    in_position = False
    entry_price = 0.0
    entry_type = None
    for i in range(len(df)):
        row = df.iloc[i]
        # Exit logic first
        if in_position:
            if entry_type == 'long':
                if row['long_exit']:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) / entry_price
                    pnls.append(pnl)
                    in_position = False
            else:  # short
                if row['short_exit']:
                    exit_price = row['close']
                    pnl = (entry_price - exit_price) / entry_price
                    pnls.append(pnl)
                    in_position = False
        # Entry logic (only if not in position)
        if not in_position:
            if row['long_entry']:
                in_position = True
                entry_type = 'long'
                entry_price = row['close']
            elif row['short_entry']:
                in_position = True
                entry_type = 'short'
                entry_price = row['close']

    print(f"Number of trades: {len(pnls)}")
    if len(pnls) > 0:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        wr = len(wins) / len(pnls)
        pf = abs(np.mean(wins) / np.mean(losses)) if np.mean(losses) != 0 else np.inf
        tr = sum(pnls)
        ex = np.mean(pnls) if pnls else 0
        # Max drawdown
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd = np.max(running_max - cum) if len(cum) > 0 else 0
        # Sharpe (approx)
        sh = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        print(f"Results:")
        print(f"  Win Rate: {wr*100:.2f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Sharpe: {sh:.2f}")
        print(f"  Max Drawdown: {dd*100:.2f}%")
        print(f"  Total Return: {tr*100:.2f}%")
        print(f"  Expectancy per trade: {ex*100:.3f}%")
        print(f"  Average Win: {np.mean(wins)*100:.2f}% if wins else 0")
        print(f"  Average Loss: {np.mean(losses)*100:.2f}% if losses else 0")
    else:
        print("No trades completed.")

# Now test on validation set quickly
print("\n--- Testing on validation set ---")
df_val = val_df.copy()
df_val['timestamp'] = pd.to_datetime(df_val['timestamp'])
df_val = df_val.set_index('timestamp').sort_index()
# Compute same indicators (we could reuse but for simplicity recompute)
# ... (same as above) ...
# For brevity, we'll just compute signals and do a quick simulation
# We'll copy the same code but for df_val
# Given time, let's just do a quick version
# Compute indicators
bb_ma = df_val['close'].rolling(params['bb_period']).mean()
bb_std_val = df_val['close'].rolling(params['bb_period']).std()
df_val['bb_upper'] = bb_ma + bb_std_val * params['bb_std']
df_val['bb_lower'] = bb_ma - bb_std_val * params['bb_std']
df_val['bb_mid'] = bb_ma
delta = df_val['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(params['rsi_period']).mean()
avg_loss = loss.rolling(params['rsi_period']).mean()
rs = avg_gain / avg_loss
df_val['rsi'] = 100 - (100 / (1 + rs))
high_low = df_val['high'] - df_val['low']
high_close = np.abs(df_val['high'] - df_val['close'].shift())
low_close = np.abs(df_val['low'] - df_val['close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_val['atr'] = tr.rolling(params['atr_period']).mean()
df_val['atr_regime'] = tr.rolling(params['atr_regime_period']).mean()
df_val['trend'] = df_val['close'].rolling(params['trend_period']).mean()
df_val['regime_filter'] = df_val['atr'] <= params['atr_regime_mult'] * df_val['atr_regime']
df_val['hour'] = df_val.index.hour
df_val['session_filter'] = (df_val['hour'] >= params['session_start']) & (df_val['hour'] <= params['session_end'])
df_val['filter_ok'] = df_val['regime_filter'] & df_val['session_filter']
df_val['long_entry'] = (df_val['close'] < df_val['bb_lower']) & (df_val['rsi'] < 30) & (df_val['close'] > df_val['trend']) & df_val['filter_ok']
df_val['short_entry'] = (df_val['close'] > df_val['bb_upper']) & (df_val['rsi'] > 70) & (df_val['close'] < df_val['trend']) & df_val['filter_ok']
df_val['long_exit'] = (df_val['close'] >= df_val['bb_mid']) | (df_val['rsi'] > 70)
df_val['short_exit'] = (df_val['close'] <= df_val['bb_mid']) | (df_val['rsi'] < 30)

# Simulate
pnls_val = []
in_position = False
entry_price = 0.0
entry_type = None
for i in range(len(df_val)):
    row = df_val.iloc[i]
    if in_position:
        if entry_type == 'long':
            if row['long_exit']:
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price
                pnls_val.append(pnl)
                in_position = False
        else:
            if row['short_exit']:
                exit_price = row['close']
                pnl = (entry_price - exit_price) / entry_price
                pnls_val.append(pnl)
                in_position = False
    if not in_position:
        if row['long_entry']:
            in_position = True
            entry_type = 'long'
            entry_price = row['close']
        elif row['short_entry']:
            in_position = True
            entry_type = 'short'
            entry_price = row['close']
print(f"Validation trades: {len(pnls_val)}")
if len(pnls_val) > 0:
    wins = [p for p in pnls_val if p > 0]
    losses = [p for p in pnls_val if p < 0]
    wr = len(wins) / len(pnls_val)
    pf = abs(np.mean(wins) / np.mean(losses)) if np.mean(losses) != 0 else np.inf
    tr = sum(pnls_val)
    ex = np.mean(pnls_val) if pnls_val else 0
    cum = np.cumsum(pnls_val)
    running_max = np.maximum.accumulate(cum)
    dd = np.max(running_max - cum) if len(cum) > 0 else 0
    sh = np.mean(pnls_val) / np.std(pnls_val) * np.sqrt(252) if np.std(pnls_val) > 0 else 0
    print(f"Validation Results:")
    print(f"  Win Rate: {wr*100:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Sharpe: {sh:.2f}")
    print(f"  Max Drawdown: {dd*100:.2f}%")
    print(f"  Total Return: {tr*100:.2f}%")
    print(f"  Expectancy per trade: {ex*100:.3f}%")
else:
    print("No validation trades.")

# Save the indicators and signals for inspection if needed
# df.to_csv('debug_indicators_train.csv')
# df_val.to_csv('debug_indicators_val.csv')
print("\nDone.")