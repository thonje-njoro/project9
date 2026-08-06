import pandas as pd
import numpy as np

df = pd.read_parquet("data/XAUUSD_1h_raw.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
df["hour"] = df.index.hour
df["date"] = df.index.date

print("=== Volume by Hour (UTC) ===")
hourly_vol = df.groupby("hour")["volume"].agg(["mean", "sum", "count"])
hourly_vol.columns = ["avg_vol", "total_vol", "bars"]
print(hourly_vol.sort_values("avg_vol", ascending=False).head(10))

print("\n=== Price Range by Hour (UTC) ===")
df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
hourly_range = df.groupby("hour")["range_pct"].agg(["mean", "std"])
hourly_range.columns = ["avg_range_pct", "std_range"]
print(hourly_range.sort_values("avg_range_pct", ascending=False).head(10))

print("\n=== Daily Stats ===")
daily = df.groupby("date").agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volume=("volume", "sum")
)
daily["daily_range_pct"] = (daily["high"] - daily["low"]) / daily["close"] * 100
daily["daily_return"] = daily["close"].pct_change() * 100
print("Date range:", daily.index[0], "to", daily.index[-1])
print("Avg daily range:", round(daily["daily_range_pct"].mean(), 2), "%")
print("Avg daily return:", round(daily["daily_return"].mean(), 3), "%")
print("Daily vol (std):", round(daily["daily_return"].std(), 2), "%")

print("\n=== First Hour Analysis (London Open 7 UTC) ===")
london_first = df[df["hour"] == 7].copy()
london_first["first_hour_range"] = (london_first["high"] - london_first["low"]) / london_first["open"] * 100
print("Avg first hour range:", round(london_first["first_hour_range"].mean(), 3), "%")
print("Median:", round(london_first["first_hour_range"].median(), 3), "%")
print("Bars:", len(london_first))

print("\n=== First Hour Analysis (NY Open 13 UTC) ===")
ny_first = df[df["hour"] == 13].copy()
ny_first["first_hour_range"] = (ny_first["high"] - ny_first["low"]) / ny_first["open"] * 100
print("Avg first hour range:", round(ny_first["first_hour_range"].mean(), 3), "%")
print("Median:", round(ny_first["first_hour_range"].median(), 3), "%")
print("Bars:", len(ny_first))

print("\n=== Relative Volume Analysis (NY Open) ===")
ny_vol = df[df["hour"] == 13][["volume"]].copy()
ny_vol["avg_14d"] = ny_vol["volume"].rolling(14).mean()
ny_vol["rel_vol"] = ny_vol["volume"] / ny_vol["avg_14d"] * 100
print("Avg relative volume at NY open:", round(ny_vol["rel_vol"].mean(), 1), "%")
print("Median:", round(ny_vol["rel_vol"].median(), 1), "%")

print("\n=== ORB Direction Bias (NY Open 13 UTC) ===")
ny_bars = df[df["hour"] == 13].copy()
ny_bars["first_candle_bullish"] = ny_bars["close"] > ny_bars["open"]

follow_count = 0
total_count = 0
long_follow = 0
long_total = 0
short_follow = 0
short_total = 0
for idx, row in ny_bars.iterrows():
    date = idx.date()
    day_data = df[(df["date"] == date) & (df.index > idx)]
    if len(day_data) < 2:
        continue
    day_close = day_data.iloc[-1]["close"]
    first_close = row["close"]
    
    if row["first_candle_bullish"]:
        long_total += 1
        if day_close > first_close:
            long_follow += 1
            follow_count += 1
    else:
        short_total += 1
        if day_close < first_close:
            short_follow += 1
            follow_count += 1
    total_count += 1

print("Total days:", total_count)
print("First candle direction followed:", follow_count, "/", total_count, "(", round(follow_count/total_count*100, 1), "%)")
print("Long signals:", long_follow, "/", long_total, "(", round(long_follow/long_total*100, 1), "%)")
print("Short signals:", short_follow, "/", short_total, "(", round(short_follow/short_total*100, 1), "%)")

print("\n=== ATR Analysis (14-period on 1h) ===")
atr = df.groupby("date").apply(lambda x: pd.Series({
    "true_range": max(
        x["high"].max() - x["low"].min(),
        abs(x["high"].max() - x["close"].iloc[-1]),
        abs(x["low"].min() - x["close"].iloc[-1])
    ),
    "close": x["close"].iloc[-1]
}))
atr["atr_pct"] = atr["true_range"] / atr["close"] * 100
atr["atr_14"] = atr["atr_pct"].rolling(14).mean()
print("Avg daily ATR:", round(atr["atr_pct"].mean(), 2), "%")
print("Avg 14-day ATR:", round(atr["atr_14"].mean(), 2), "%")
print("10% ATR stop:", round(atr["atr_14"].mean() * 0.1, 3), "%")

print("\n=== Relative Volume vs Returns ===")
ny_vol_dates = set(ny_vol[ny_vol["rel_vol"] > 100].index.date)
high_vol_returns = []
low_vol_returns = []
for idx, row in ny_bars.iterrows():
    date = idx.date()
    day_data = df[(df["date"] == date) & (df.index > idx)]
    if len(day_data) < 2:
        continue
    entry_price = row["open"]
    exit_price = day_data.iloc[-1]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    if date in ny_vol_dates:
        high_vol_returns.append(ret)
    else:
        low_vol_returns.append(ret)

print("High rel vol days:", len(high_vol_returns), "avg return:", round(np.mean(high_vol_returns), 3), "%")
print("Low rel vol days:", len(low_vol_returns), "avg return:", round(np.mean(low_vol_returns), 3), "%")
