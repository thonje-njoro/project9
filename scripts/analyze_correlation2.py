import pandas as pd
import numpy as np
import os

TS = "/home/admin1/project9/trading-system"
BT = "/home/admin1/project9/backtest"

xau = pd.read_parquet(os.path.join(TS, "data/XAUUSD_1h_raw.parquet"))
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

# Load GLD from backtest cache
for inst in ["GLD", "GDX", "SLV", "CPER", "TLT", "SPY", "QQQ", "TSLA", "IWM"]:
    try:
        path = os.path.join(BT, f"data/cache/{inst}_2022-01-01_2024-12-31.parquet")
        if not os.path.exists(path):
            # try other patterns
            found = False
            for f in os.listdir(os.path.join(BT, "data/cache")):
                if f.startswith(inst + "_") and f.endswith(".parquet"):
                    path = os.path.join(BT, "data/cache", f)
                    found = True
                    break
            if not found:
                continue
        
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        
        # Handle both column naming conventions
        close_col = "Close" if "Close" in df.columns else "close"
        df_daily = df.groupby(df.index.date).agg(close=(close_col, "last"))
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily[f"{inst}_return"] = df_daily["close"].pct_change()
        
        merged = pd.merge(xau_daily[["xau_return"]], df_daily[[f"{inst}_return"]], 
                          left_index=True, right_index=True, how="inner")
        corr = merged["xau_return"].corr(merged[f"{inst}_return"])
        
        # Check lag correlation (does GLD lead XAU?)
        merged[f"{inst}_lag1"] = merged[f"{inst}_return"].shift(1)
        lag_corr = merged["xau_return"].corr(merged[f"{inst}_lag1"])
        
        print(f"XAUUSD vs {inst}: corr={round(corr, 4)} | lag1_corr={round(lag_corr, 4)} | days={len(merged)}")
    except Exception as e:
        print(f"{inst} error: {e}")

# SPY on 1min data for ORB analysis
print("\n=== SPY 1-min Data (for ORB reference) ===")
try:
    spy = pd.read_parquet(os.path.join(BT, "data/cache/SPY_2024-01-01_2024-12-31.parquet"))
    spy.index = pd.to_datetime(spy.index)
    spy["hour"] = spy.index.hour
    spy["minute"] = spy.index.minute
    spy["date"] = spy.index.date
    
    # First 5 minutes of trading (9:30-9:35 ET = 14:30-14:35 UTC)
    spy_orb = spy[(spy["hour"] == 14) & (spy["minute"] >= 30) & (spy["minute"] < 35)]
    print("SPY 5-min ORB bars:", len(spy_orb))
    print("Unique dates:", spy_orb["date"].nunique())
    
    # Direction bias
    orb_agg = spy_orb.groupby("date").agg(
        orb_open=("Open", "first"),
        orb_close=("Close", "last"),
        orb_high=("High", "max"),
        orb_low=("Low", "min")
    )
    orb_agg["direction"] = np.where(orb_agg["orb_close"] > orb_agg["orb_open"], 1, -1)
    
    # Check end of day
    eod = spy.groupby("date").agg(eod_close=("Close", "last"))
    merged = pd.merge(orb_agg, eod, left_index=True, right_index=True)
    
    long_correct = len(merged[(merged["direction"] == 1) & (merged["eod_close"] > merged["orb_close"])])
    long_total = len(merged[merged["direction"] == 1])
    short_correct = len(merged[(merged["direction"] == -1) & (merged["eod_close"] < merged["orb_close"])])
    short_total = len(merged[merged["direction"] == -1])
    
    total_correct = long_correct + short_correct
    total = long_total + short_total
    
    print(f"SPY ORB direction persistence: {total_correct}/{total} ({round(total_correct/total*100, 1)}%)")
    print(f"Long: {long_correct}/{long_total} ({round(long_correct/long_total*100, 1)}%)")
    print(f"Short: {short_correct}/{short_total} ({round(short_correct/short_total*100, 1)}%)")
    
except Exception as e:
    print(f"SPY error: {e}")

# QQQ on 1min data
print("\n=== QQQ 1-min Data ===")
try:
    qqq = pd.read_parquet(os.path.join(BT, "data/cache/QQQ_2024-01-01_2024-12-31.parquet"))
    qqq.index = pd.to_datetime(qqq.index)
    qqq["hour"] = qqq.index.hour
    qqq["minute"] = qqq.index.minute
    qqq["date"] = qqq.index.date
    
    qqq_orb = qqq[(qqq["hour"] == 14) & (qqq["minute"] >= 30) & (qqq["minute"] < 35)]
    orb_agg = qqq_orb.groupby("date").agg(
        orb_open=("Open", "first"),
        orb_close=("Close", "last"),
    )
    orb_agg["direction"] = np.where(orb_agg["orb_close"] > orb_agg["orb_open"], 1, -1)
    eod = qqq.groupby("date").agg(eod_close=("Close", "last"))
    merged = pd.merge(orb_agg, eod, left_index=True, right_index=True)
    
    long_correct = len(merged[(merged["direction"] == 1) & (merged["eod_close"] > merged["orb_close"])])
    long_total = len(merged[merged["direction"] == 1])
    short_correct = len(merged[(merged["direction"] == -1) & (merged["eod_close"] < merged["orb_close"])])
    short_total = len(merged[merged["direction"] == -1])
    
    total_correct = long_correct + short_correct
    total = long_total + short_total
    
    print(f"QQQ ORB direction persistence: {total_correct}/{total} ({round(total_correct/total*100, 1)}%)")
    print(f"Long: {long_correct}/{long_total} ({round(long_correct/long_total*100, 1)}%)")
    print(f"Short: {short_correct}/{short_total} ({round(short_correct/short_total*100, 1)}%)")
    
except Exception as e:
    print(f"QQQ error: {e}")
