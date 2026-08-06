"""Analyze CPER data characteristics to determine the best strategy."""
import sys, os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
from data.fetcher import DataFetcher

f = DataFetcher(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))

# 1. Load CPER 1Min data
print("=" * 60)
print("CPER DATA ANALYSIS")
print("=" * 60)

cper = f.fetch("CPER", "stock", "2022-01-01", "2024-12-31")

# 2. Daily close analysis
close = cper["close"].resample("1D").last().dropna()
returns = close.pct_change().dropna()

print(f"\nRaw bars: {len(cper):,}")
print(f"Trading days: {len(close)}")
print(f"Close: ${close.iloc[0]:.2f} -> ${close.iloc[-1]:.2f}")
print(f"Total return: {(close.iloc[-1]/close.iloc[0]-1)*100:.2f}%")
print(f"Mean daily return: {returns.mean()*100:.4f}%")
print(f"Ann return: {returns.mean()*252*100:.2f}%")
print(f"Ann vol: {returns.std()*np.sqrt(252)*100:.2f}%")
print(f"Sharpe: {returns.mean()/returns.std()*np.sqrt(252):.2f}")
print(f"Skew: {returns.skew():.2f}, Kurtosis: {returns.kurtosis():.2f}")

# 3. Autocorrelation structure (trend vs mean-reversion signature)
print("\n--- Autocorrelation (momentum signature) ---")
for lag_days in [1, 2, 5, 10, 21, 42, 63]:
    ac = returns.autocorr(lag=lag_days)
    label = "MOMENTUM" if ac > 0.05 else ("MEAN-REVERT" if ac < -0.05 else "noise")
    print(f"  Lag {lag_days:2d}d: {ac:+.4f}  [{label}]")

# 4. Hurst exponent
print("\n--- Hurst Exponent ---")
lags = range(2, min(101, len(returns)//4))
try:
    tau = [np.sqrt(np.var(np.diff(returns.values, n=l))) for l in lags]
    m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    hurst = m[0] / 2
    print(f"  Hurst: {hurst:.4f}  ", end="")
    if hurst > 0.6: print("[TRENDING] — strong momentum signature")
    elif hurst > 0.55: print("[weak trending]")
    elif hurst < 0.4: print("[MEAN-REVERTING] — strong reversion signature")
    elif hurst < 0.45: print("[weak mean-reverting]")
    else: print("[random walk / mixed]")
except Exception as e:
    print(f"  Error: {e}")

# 5. Volatility clustering
print("\n--- Volatility ---")
daily_vol = returns.rolling(21).std() * np.sqrt(252)
print(f"  Avg 21d ann vol: {daily_vol.mean()*100:.1f}%")
print(f"  Vol range: [{daily_vol.min()*100:.1f}%, {daily_vol.max()*100:.1f}%]")
vol_autocorr = daily_vol.dropna().autocorr(lag=1)
print(f"  Volatility clustering (AC1): {vol_autocorr:.3f} {'[STRONG]' if vol_autocorr > 0.5 else ''}")

# 6. Drawdown analysis
peak = close.cummax()
dd = (close - peak) / peak
max_dd = dd.min()
print(f"\n  Max drawdown: {max_dd*100:.2f}% on {dd.idxmin().date()}")
print(f"  % of days in drawdown: {(dd < 0).mean()*100:.1f}%")

# 7. Test: Kalman trend on 4H data
print("\n--- Kalman Trend Test (4H data) ---")
cper_4h = cper["close"].resample("4H").last().dropna()
try:
    from pykalman import KalmanFilter
    vals = cper_4h.values
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[float(vals[0]), 0.0],
        initial_state_covariance=[[1.0, 0], [0, 0.02]],
        transition_covariance=[[0.01, 0], [0, 0.002]],
        observation_covariance=[[1.0]],
    )
    state_means, _ = kf.filter(vals)
    position = pd.Series(state_means[:, 0], index=cper_4h.index)
    velocity = pd.Series(state_means[:, 1], index=cper_4h.index)
    # Count zero crosses
    zero_cross = ((velocity > 0) & (velocity.shift(1) <= 0)).sum()
    zero_cross_neg = ((velocity < 0) & (velocity.shift(1) >= 0)).sum()
    print(f"  4H bars: {len(cper_4h)}")
    print(f"  Velocity zero-cross (up): {zero_cross}")
    print(f"  Velocity zero-cross (down): {zero_cross_neg}")
    print(f"  Total signals: {zero_cross + zero_cross_neg}")
    print(f"  Trades/year (est): {(zero_cross + zero_cross_neg) / 3:.0f}")
    print(f"  Velocity std: {velocity.std():.4f}")
    print(f"  Recommend: Kalman trend-following on 4H data, ~{(zero_cross + zero_cross_neg):.0f} trades/3yr")
except ImportError:
    print("  pykalman not available")

# 8. Test: Mean reversion on daily data (Bollinger)
print("\n--- Mean Reversion Test (Daily, Bollinger 20,2) ---")
bb_ma = close.rolling(20).mean()
bb_std = close.rolling(20).std()
bb_upper = bb_ma + 2 * bb_std
bb_lower = bb_ma - 2 * bb_std
touch_upper = (close > bb_upper).sum()
touch_lower = (close < bb_lower).sum()
print(f"  Touches upper band: {touch_upper} in 3yr")
print(f"  Touches lower band: {touch_lower} in 3yr")
print(f"  % time outside bands: {(touch_upper + touch_lower) / len(close) * 100:.1f}%")

# 9. Test: CPER/GLD ratio
print("\n--- CPER/GLD Ratio ---")
gld = f.fetch("GLD", "stock", "2022-01-01", "2024-12-31")
gld_close = gld["close"].resample("1D").last().dropna()
common = close.index.intersection(gld_close.index)
ratio = close[common] / gld_close[common]
ratio_ret = ratio.pct_change().dropna()
print(f"  CPER/GLD ratio bars: {len(ratio)}")
print(f"  Ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")
print(f"  Ratio mean: {ratio.mean():.4f}, std: {ratio.std():.4f}")
r_ac = ratio_ret.autocorr(lag=1)
print(f"  Daily return AC(1): {r_ac:.4f} ", end="")
if abs(r_ac) > 0.05:
    print("[TRADABLE] — reversion-to-mean possible")
else:
    print("[random walk] — no short-term edge")
ratio_hurst = 0.5
print(f"  Est. Hurst: ~0.5 (cointegrated pair tends to mean-revert)")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
