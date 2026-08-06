import pandas as pd
import numpy as np

# Load XAUUSD daily data
xau = pd.read_parquet("data/XAUUSD_1h_raw.parquet")
xau["timestamp"] = pd.to_datetime(xau["timestamp"])
xau = xau.set_index("timestamp")
xau_daily = xau.groupby(xau.index.date).agg(
    open=("open", "first"),
    close=("close", "last"),
    high=("high", "max"),
    low=("low", "min"),
    volume=("volume", "sum")
)
xau_daily.index = pd.to_datetime(xau_daily.index)
xau_daily["xau_return"] = xau_daily["close"].pct_change()

print("=== XAUUSD Daily Stats ===")
print("Date range:", xau_daily.index[0], "to", xau_daily.index[-1])
print("Total return:", round((xau_daily["close"].iloc[-1] / xau_daily["close"].iloc[0] - 1) * 100, 1), "%")

# Load GLD
try:
    gld = pd.read_parquet("data/cache/GLD_2022-01-01_2024-12-31.parquet")
    gld.index = pd.to_datetime(gld.index)
    gld_daily = gld.groupby(gld.index.date).agg(
        open=("Open", "first"),
        close=("Close", "last")
    )
    gld_daily.index = pd.to_datetime(gld_daily.index)
    gld_daily["gld_return"] = gld_daily["close"].pct_change()
    
    # Merge and correlate
    merged = pd.merge(xau_daily[["xau_return"]], gld_daily[["gld_return"]], 
                      left_index=True, right_index=True, how="inner")
    corr = merged["xau_return"].corr(merged["gld_return"])
    print("\n=== XAUUSD vs GLD Correlation ===")
    print("Correlation:", round(corr, 4))
    print("Overlapping days:", len(merged))
    
    # Check if GLD leads XAUUSD (1-day lag)
    merged["gld_lag1"] = merged["gld_return"].shift(1)
    lag_corr = merged["xau_return"].corr(merged["gld_lag1"])
    print("GLD leads XAUUSD (1-day lag) corr:", round(lag_corr, 4))
    
except Exception as e:
    print("GLD error:", e)

# Check other instruments
for inst in ["GDX", "SLV", "CPER", "TLT"]:
    try:
        path = f"data/cache/{inst}_2022-01-01_2024-12-31.parquet"
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df_daily = df.groupby(df.index.date).agg(close=("Close", "last"))
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily[f"{inst}_return"] = df_daily["close"].pct_change()
        
        merged = pd.merge(xau_daily[["xau_return"]], df_daily[[f"{inst}_return"]], 
                          left_index=True, right_index=True, how="inner")
        corr = merged["xau_return"].corr(merged[f"{inst}_return"])
        print(f"\nXAUUSD vs {inst} Correlation: {round(corr, 4)} ({len(merged)} days)")
    except Exception as e:
        print(f"{inst} error:", e)

# Session-based analysis for XAUUSD
print("\n=== Session Returns (XAUUSD) ===")
xau["hour"] = xau.index.hour
xau["date"] = xau.index.date

# London session: 7-16 UTC
london = xau[(xau["hour"] >= 7) & (xau["hour"] < 16)]
london_ret = london.groupby("date").apply(lambda x: (x["close"].iloc[-1] - x["open"].iloc[0]) / x["open"].iloc[0] * 100)
print("London session avg return:", round(london_ret.mean(), 3), "%")
print("London session vol:", round(london_ret.std(), 3), "%")

# NY session: 13-22 UTC
ny = xau[(xau["hour"] >= 13) & (xau["hour"] < 22)]
ny_ret = ny.groupby("date").apply(lambda x: (x["close"].iloc[-1] - x["open"].iloc[0]) / x["open"].iloc[0] * 100)
print("NY session avg return:", round(ny_ret.mean(), 3), "%")
print("NY session vol:", round(ny_ret.std(), 3), "%")

# Asia session: 0-7 UTC
asia = xau[(xau["hour"] >= 0) & (xau["hour"] < 7)]
asia_ret = asia.groupby("date").apply(lambda x: (x["close"].iloc[-1] - x["open"].iloc[0]) / x["open"].iloc[0] * 100)
print("Asia session avg return:", round(asia_ret.mean(), 3), "%")
print("Asia session vol:", round(asia_ret.std(), 3), "%")

# First 5-min ORB equivalent: Use first hour of NY session
print("\n=== NY First Hour ORB (13 UTC) ===")
ny_first = xau[xau["hour"] == 13].copy()
ny_first["direction"] = np.where(ny_first["close"] > ny_first["open"], 1, -1)
ny_first["range"] = (ny_first["high"] - ny_first["low"]) / ny_first["open"] * 100

# Check if direction persists for next 4 hours
correct = 0
total = 0
for idx, row in ny_first.iterrows():
    date = idx.date()
    rest = xau[(xau["date"] == date) & (xau["hour"] > 13)]
    if len(rest) < 2:
        continue
    end_price = rest.iloc[-1]["close"]
    start_price = row["close"]
    
    if row["direction"] == 1:  # bullish first candle
        if end_price > start_price:
            correct += 1
    else:  # bearish first candle
        if end_price < start_price:
            correct += 1
    total += 1

print("Direction persistence (next 4h):", correct, "/", total, "(", round(correct/total*100, 1), "%)")

# High range days vs low range days
median_range = ny_first["range"].median()
high_range = ny_first[ny_first["range"] > median_range]
low_range = ny_first[ny_first["range"] <= median_range]

print("\nHigh range days (>median):", len(high_range))
print("Low range days (<=median):", len(low_range))

# Check returns on high range days
high_range_returns = []
low_range_returns = []
for idx, row in ny_first.iterrows():
    date = idx.date()
    rest = xau[(xau["date"] == date) & (xau["hour"] > 13)]
    if len(rest) < 2:
        continue
    end_price = rest.iloc[-1]["close"]
    start_price = row["open"]
    ret = (end_price - start_price) / start_price * 100
    
    if row["range"] > median_range:
        high_range_returns.append(ret)
    else:
        low_range_returns.append(ret)

print("High range avg return:", round(np.mean(high_range_returns), 3), "%")
print("Low range avg return:", round(np.mean(low_range_returns), 3), "%")
