import os
import pandas as pd
from dotenv import load_dotenv
from lse import LSE

load_dotenv()
client = LSE(api_key=os.getenv("LSE_API_KEY"))

SYMBOL = "XAU/USD"          # change to your instrument
TIMEFRAME = "1h"           # good starting resolution: not too heavy, not too coarse
START = "2019-01-01"
END = "2026-01-01"

def fetch_all(symbol, timeframe, start, end, page_days=180):
    """Page through the vault in chunks since one call is capped by plan row limit."""
    all_rows = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while cursor < end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=page_days), end_ts)
        rows = client.candles(
            symbol, timeframe,
            start=cursor.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            limit=5000,
        )
        print(f"  {cursor.date()} -> {chunk_end.date()}: {len(rows)} rows")
        all_rows.extend(rows)
        cursor = chunk_end

    return pd.DataFrame(all_rows)

print(f"Fetching {SYMBOL} {TIMEFRAME} from {START} to {END}...")
df = fetch_all(SYMBOL, TIMEFRAME, START, END)
df = df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
print(f"Total rows: {len(df)}")

# Save the full raw pull
raw_path = f"data/{SYMBOL.replace('/', '')}_{TIMEFRAME}_raw.parquet"
df.to_parquet(raw_path)
print(f"Saved raw data to {raw_path}")

# Chronological split: train / validation / holdout
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
holdout = df.iloc[val_end:]

train.to_parquet(f"data/{SYMBOL.replace('/', '')}_{TIMEFRAME}_train.parquet")
val.to_parquet(f"data/{SYMBOL.replace('/', '')}_{TIMEFRAME}_val.parquet")
holdout.to_parquet(f"data/{SYMBOL.replace('/', '')}_{TIMEFRAME}_holdout.parquet")

print(f"Train: {len(train)} rows ({train['timestamp'].min()} -> {train['timestamp'].max()})")
print(f"Val:   {len(val)} rows ({val['timestamp'].min()} -> {val['timestamp'].max()})")
print(f"Holdout: {len(holdout)} rows ({holdout['timestamp'].min()} -> {holdout['timestamp'].max()})")
