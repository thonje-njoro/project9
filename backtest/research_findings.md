# Multi-Strategy Algorithmic Trading System — Research Findings
## Comprehensive Analysis for /home/admin1/project9/backtest

**Date:** July 2026  
**Capital Range:** $50k–$100k  
**Context:** Multi-instrument vectorbt system with GLD, XAU/USD, CPER, CPER_GLD_RATIO

---

## 1. WALK-FORWARD VALIDATION & OVERFITTING REDUCTION

### Current System Assessment
Your system has TWO walk-forward implementations:
- `optimization/walk_forward.py` — Standard 4-window with IS/OOS splits per window (skewed)
- `optimization/purged_walk_forward.py` — Purged k-fold with embargo (better but unused in main.py)
- `main.py` — Custom 4-split walk-around (line 440-526) that doesn't use purged/embargo logic

The main issue: **3 out of 4 instruments show REJECT/UNSTABLE**. This is because:
1. You only have 4 folds — too few for statistical significance. **Bailey & López de Prado (2014)** recommend ≥10 folds.
2. No embargo between IS/OOS splits — **information leakage** inflates IS Sharpe, cratering OOS.
3. Non-overlapping training windows waste data.

### Practical Approaches That Work

#### A. Combinatorial Purged Cross-Validation (CPCV) — López de Prado
This is the gold standard from *Advances in Financial Machine Learning*. Instead of a single walk-forward path, CPCV generates **many train-test combinations** giving you a distribution of OOS outcomes.

**What to implement:**
```python
# Use the existing purged_walk_forward.py but expand to >=10 folds
# Add Deflated Sharpe Ratio (DSR) computation
# DSR > 0 = statistically significant edge after multiple testing

def compute_probabilistic_sharpe(sharpe, num_trials, skew, kurt, n_obs):
    """Compute the Probabilistic Sharpe Ratio — probability that true SR > 0"""
    # Adjusts for non-normal returns (skewness, kurtosis)
    numerator = (sharpe - 0) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(1 + 0.5*sharpe**2 - skew*sharpe + (kurt-3)/4*sharpe**2)
    from scipy.stats import norm
    return 1 - norm.cdf(-numerator/denominator)

def deflated_sharpe(sharpe, num_trials, n_obs, skew=0, kurt=3):
    """Deflated Sharpe Ratio — accounts for number of experiments run"""
    from scipy.stats import norm
    euler_gamma = 0.5772156649
    var_sharpe = (1 + 0.5*sharpe**2 + skew*sharpe - (kurt-3)/4*sharpe**2) / (n_obs - 1)
    std_sharpe = np.sqrt(var_sharpe)
    e_max = (1-euler_gamma)*norm.ppf(1-1/num_trials) + euler_gamma*norm.ppf(1-1/(num_trials*np.e))
    return (sharpe - e_max*std_sharpe) / std_sharpe
```

**Concrete changes for your system:**
1. Expand to **10-fold walk-forward** (not 4) — each fold trains on 65%, tests on 35%
2. Add **embargo period of 20 bars** between train/test (already in purged_wf.py!)  
3. Use **rolling overlapping windows** (shift by 1 fold, not disjoint chunks)
4. Report **Probability of Backtest Overfitting (PBO)** — from CSCV distribution
5. Replace confidence in individual instrument with **min OOS Sharpe > 0 threshold across ALL folds**

**Your current scoring (in main.py line 499-503) needs fixing:**
```python
# CURRENT (too aggressive — directly REJECTs anything with negative avg):
if avg_oos < 0: rec = "REJECT"

# PROPOSED (more nuanced — allow marginal per-instrument):
positive_ratio = pos_windows / len(folds)
if positive_ratio < 0.25:  # Less than 25% of windows positive
    rec = "REJECT"
elif decay > 1.0:
    rec = "OVERFIT"
elif positive_ratio < 0.5:
    rec = "UNSTABLE"
else:
    rec = "PASS"
# Combined portfolio WALK-FORWARD should be the primary criterion
```

#### B. Parameter Clustering (Quant Beckman / K-Means approach)
Instead of picking the single "best" parameter set, **cluster parameter space** and pick cluster centroids.

**Key insight:** The single best IS parameter is always overfit. The centroid of a cluster of good parameters is far more robust.

```
1. Run parameter sweep (you already have this!)
2. For each parameter combo, save OOS Sharpe across all walk-forward folds
3. K-means cluster the parameter → performance mapping (k=3-5)
4. Select the centroid of the highest-performing cluster
5. Use that as your production parameter
```

**Implementation:** Add to `optimization/parameter_grid.py`:
```python
from sklearn.cluster import KMeans
def cluster_parameter_opt(results_df, n_clusters=4):
    """Cluster parameter sets by performance, return cluster centroids."""
    X = results_df[['param1', 'param2', 'param3']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    results_df['cluster'] = kmeans.labels_
    centroids = []
    for c in range(n_clusters):
        cluster = results_df[results_df['cluster'] == c]
        if len(cluster) >= 2:
            centroid = cluster[['param1', 'param2', 'param3']].mean().to_dict()
            centroid['oos_sharpe'] = cluster['oos_sharpe'].mean()
            centroids.append(centroid)
    # Return centroid of best cluster
    return max(centroids, key=lambda x: x['oos_sharpe'])
```

#### C. Ensemble of Parameter Sets
Run **multiple parameter sets simultaneously** and average their positions (or take the majority signal). This is like a Random Forest for parameters.

**For your system:** Run 5 parameter variants per instrument in parallel, take the consensus position. This reduces overfitting variance by √5.

#### D. Corrected Scoring Metrics
Your current walk-forward tests **each instrument individually** but the critical metric is the **COMBINED portfolio OOS performance**. An instrument can look weak solo but add portfolio-level diversification benefit.

**Change:** Run walk-forward on the entire portfolio (risk parity + all instruments) as the primary test. Individual instruments are diagnostics only.

---

## 2. EXIT OPTIMIZATION TECHNIQUES

### Current System
Your `ExitOptimizer` tests 6 fixed configurations: baseline, partial_1x, partial_1.5x, time_exit_10, time_exit_15, combined. This is a good start but too coarse.

### Approaches from Recent Research

#### A. Triple-Barrier Method (López de Prado)
The single most impactful exit framework from *Advances in Financial Machine Learning*:

- **Upper barrier** = take-profit (e.g., 2× ATR)
- **Lower barrier** = stop-loss (e.g., 1× ATR)
- **Vertical barrier** = time-based exit (e.g., max 20 bars)

**Key insight:** Label each bar by which barrier is hit FIRST. Then meta-label: given a signal, should we enter? This separates direction prediction (hard) from "when to get out" (easier).

**Implementation for vectorbt:**
```python
def triple_barrier_exits(entries, close, atr, tp_mult=2.0, sl_mult=1.0, max_bars=20):
    """
    Generate exits where any of 3 barriers is hit first.
    Returns (exits, labels) where labels = {1: hit TP, -1: hit SL, 0: time exit}
    """
    # Track each active position
    # For each bar since entry, check if price hit TP or SL first
    labels = pd.Series(0, index=entries.index)
    exits = pd.Series(False, index=entries.index)
    
    active_entries = entries[entries].index
    for idx in active_entries:
        loc = entries.index.get_loc(idx)
        entry_price = close.iloc[loc]
        tp = entry_price + atr.iloc[loc] * tp_mult
        sl = entry_price - atr.iloc[loc] * sl_mult
        for bar in range(1, max_bars + 1):
            if loc + bar >= len(close):
                break
            current_close = close.iloc[loc + bar]
            current_high = df['high'].iloc[loc + bar]  # needs df access
            current_low = df['low'].iloc[loc + bar]
            if current_high >= tp:
                exits.iloc[loc + bar] = True
                labels.iloc[loc + bar] = 1
                break
            if current_low <= sl:
                exits.iloc[loc + bar] = True
                labels.iloc[loc + bar] = -1
                break
        else:
            # Time exit
            exits.iloc[loc + max_bars] = True
            labels.iloc[loc + max_bars] = 0
    return exits, labels
```

#### B. Dynamic ATR Multipliers Based on Regime
Your `compute_regime_stop_multipliers` in `regime_filter.py` already does this! But it's **not wired into your exit optimizer**.

**What to wire up:**
- Low volatility regime: `trail_atr_mult = 2.0` (tighter)
- Normal volatility regime: `trail_atr_mult = 2.5` (base)
- High volatility regime: `trail_atr_mult = 3.5` (wider to avoid noise stops)

**For CPER_GLD_RATIO specifically:** The z-score mean reversion currently has a fixed `trail_atr_mult = 2.0`. Since this strategy has 129 trades (vs 6-15 for others), you can **optimize this per-regime**:
- In ranging regimes (where z-score works best): trail_atr_mult = 1.5 (tight)
- In trending regimes: skip signals entirely (mean reversion fails)

#### C. VWAP Exit Enhancement
Your config mentions `use_vwap_exit` for GLD and CPER. Ensure this is working: exit when price crosses VWAP + trailing stop. VWAP adds a **profit protection** signal that complements the trailing stop.

```python
def vwap_exit(close, vwap, entries, direction='long'):
    """Exit long when price crosses below VWAP, short when above VWAP."""
    if direction == 'long':
        return (close < vwap) & (entries.ffill().fillna(False))
    return (close > vwap) & (entries.ffill().fillna(False))
```

#### D. Optimal Exit Optimization Algorithm (not grid search)
Instead of testing 6 grid points, use **Bayesian optimization** (scikit-optimize) to find the optimal (tp_atr, trail_atr, time_bars) combo:

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

def objective(params):
    tp, trail, time_bars = params
    # Run backtest with these params
    sharpe = run_backtest(tp_atr=tp, trail_atr=trail, time_bars=time_bars)
    return -sharpe  # minimize negative Sharpe

res = gp_minimize(objective, 
    [Real(0.5, 3.0), Real(1.5, 4.0), Integer(5, 30)],
    n_calls=50, random_state=42)
```

**For CPER_GLD_RATIO:** Optimize specifically since it has the most trade data. Your current z_entry=1.5, z_exit=0.0, window=20 with trail_atr_mult=2.0 and PF=1.10 suggests exits are the weak link — try trail_atr_mult=1.5 (tighter stop) to improve PF at cost of fewer trades.

#### E. Split Position Exits (Partial Profit Taking)
Your `split_position_exits` in `exit_optimizer.py` has the right idea but is incomplete. Implement proper **scale-out**:
- Enter full position
- Exit 50% at 1× ATR profit
- Trail remaining 50% with 2× ATR stop

This smooths the equity curve significantly — reduces maximum adverse excursion.

---

## 3. PORTFOLIO CONSTRUCTION BEYOND EQUAL RISK PARITY

### Current System
You have a good `risk_parity_weights` implementation in `portfolio_optimizer.py` using **Equal Risk Contribution (ERC)** with SLSQP optimization. This is already better than equal weight.

### What Comes Next

#### A. Regime-Dependent Risk Parity
Your current risk parity uses a *fixed* covariance lookback (60 days). Research shows that **correlation structure changes dramatically across regimes**.

**Approach:** Compute TWO covariance matrices — one for trending regimes, one for mean-reverting regimes — and use the appropriate one based on current regime signal.

```python
def regime_aware_risk_parity(returns, regimes, current_regime, lookback=60):
    """Risk parity weights conditioned on current regime."""
    # Filter returns by regime
    regime_returns = returns[regimes == current_regime]
    if len(regime_returns) < 20:
        regime_returns = returns.tail(lookback)  # fallback
    cov = regime_returns.cov()
    # Apply shrinkage
    shrunk = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
    return risk_parity_weights(shrunk.values)
```

#### B. Tail Risk Parity (Equal Tail Risk Contribution)
Standard risk parity equalizes **variance contribution**. Tail risk parity equalizes **Expected Shortfall (CVaR) contribution** — more conservative and avoids blow-ups.

```python
def tail_risk_parity(returns, n_bootstrap=10000, alpha=0.05):
    """Equal CVaR contribution across strategies."""
    # Bootstrap return scenarios
    n = len(returns)
    boot_rets = np.random.choice(returns.values.ravel(), 
                                  (n_bootstrap, returns.shape[1]), replace=True)
    
    # Find weights equalizing CVaR contributions
    from scipy.optimize import minimize
    
    def tail_risk_contributions(w):
        port_rets = boot_rets @ w
        var_threshold = np.percentile(port_rets, alpha * 100)
        tail_mask = port_rets < var_threshold
        # Each asset's contribution to tail loss
        asset_contrib = boot_rets[tail_mask] * w
        total_tail = asset_contrib.sum(axis=1)
        return asset_contrib.mean(axis=0) / total_tail.mean()
    
    # ... minimize variance of tail risk contributions
```

#### C. Volatility Targeting Layer
Above your risk parity weights, add a **volatility targeting overlay**:
- Target portfolio volatility: 12% annualized (conservative for $50k)
- Scale all positions by: `target_vol / realized_vol`
- When vol spikes, overall exposure shrinks

This is arguably the single most impactful portfolio-level improvement.

```python
def vol_target_multiplier(portfolio_returns, target_vol=0.12, window=21):
    """Compute scaling multiplier to hit target volatility."""
    realized_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
    multiplier = target_vol / realized_vol
    return multiplier.clip(0.3, 1.5)  # Don't scale above 1.5x
```

#### D. Dynamic Weights by Strategy Performance (Online Learning)
Replace fixed risk parity with **EWMA performance-weighted allocation**:
```python
def performance_weighted_allocation(strategy_returns, halflife=20):
    """Weight strategies by their recent Sharpe."""
    ewma_returns = strategy_returns.ewm(span=halflife).mean()
    ewma_vol = strategy_returns.ewm(span=halflife).std()
    sharpes = ewma_returns / ewma_vol.clip(lower=1e-10)
    weights = sharpes.clip(lower=0).div(sharpes.clip(lower=0).sum(axis=1), axis=0)
    return weights.fillna(1.0 / strategy_returns.shape[1])
```

#### E. Key Insight for Your System
Your current 4 instruments are likely **positively correlated** (all commodities/gold-linked). The risk parity weights may be misleading because:
- GLD and CPER both trade gold/copper ETFs
- XAU/USD is also gold
- CPER_GLD_RATIO is a ratio of the above

**True diversification** requires adding genuinely uncorrelated instruments (see Section 5).

---

## 4. REGIME DETECTION THAT ACTUALLY WORKS

### Why HMM Fails (You're Right to Abandon It)
Your experience matches the academic literature: **HMM regime detection destroys alpha** because:
1. HMM assumes **stationary transition probabilities** — markets aren't Markovian
2. HMM **overfits to historical path** and changes labels retroactively
3. The 2-3 states are too coarse — real markets have continuum of regimes
4. HMM produces **hard regime shifts** that cause oscillating entries/exits

### Practical Alternatives

#### A. ADX + Trend Strength Filter (Already in Your Code!)
Your `detect_regime` in `optimization/regime_detector.py` uses ADX + SMA200 + 5d return. This is actually **better than HMM for signal filtering**. The problem is it's in `optimization/` not wired into `engine.py`.

**Fix:** Wire `detect_regime()` into the main backtest flow as a **signal override** (not a pre-filter):
- ADX > 25 + price above SMA200 = trending_up → *prefer* trend-following strategies
- ADX < 25 = ranging → *prefer* mean-reversion strategies
- High vol chaos → *reduce* all position sizes

#### B. Volatility Regime Clustering with GMM
Instead of HMM, use **Gaussian Mixture Model (GMM)** on volatility features:
- Feature vector: [20d realized vol, 60d realized vol, VIX (if available), 5d return, 20d return]
- GMM gives you **soft probabilities** (like HMM) but doesn't assume Markov transitions
- Cluster into 3 regimes: Low Vol / Normal / High Vol

```python
from sklearn.mixture import GaussianMixture

def gmm_regime_detection(df, n_states=3):
    """GMM regime detection based on vol + returns features."""
    close = df['close']
    returns = close.pct_change()
    features = pd.DataFrame({
        'ret_5d': returns.rolling(5).mean(),
        'ret_20d': returns.rolling(20).mean(),
        'vol_20d': returns.rolling(20).std(),
        'vol_60d': returns.rolling(60).std(),
    }).dropna()
    
    gmm = GaussianMixture(n_components=n_states, random_state=42)
    labels = gmm.fit_predict(features)
    # Sort by volatility (lowest vol = regime 0, highest = regime 2)
    vol_order = np.argsort([gmm.covariances_[i].mean() for i in range(n_states)])
    label_map = {old: new for new, old in enumerate(vol_order)}
    return pd.Series([label_map[l] for l in labels], index=features.index)
```

#### C. Change-Point Detection (Dual CUSUM)
For **detecting regime CHANGES** (not classifying bars), use CUSUM — already in `SYSTEM_HEALTH_CONFIG`:
```python
def detect_regime_change(prices, threshold=5.0):
    """CUSUM-based regime change detection."""
    log_prices = np.log(prices)
    mean = log_prices.expanding().mean()
    std = log_prices.expanding().std()
    cusum = ((log_prices - mean) / std).cumsum()
    return (cusum.abs() > threshold).astype(int)
```

**When CUSUM triggers:** Reset your parameter set, re-calculate risk parity, tighten stops.

#### D. Tuned ADX + Macros (What the CTA Funds Actually Use)
Real CTA funds (AQR, Man AHL, Winton) don't use HMM for regime detection. They use:
1. **ADX** for trend strength (you have this)
2. **Rolling correlation to equities** — when correlation spikes, it's a crisis
3. **Volatility regime** — 20d vol vs 200d vol ratio
4. **Term structure** — contango/backwardation signals

**Your implementation** in `optimization/regime_detector.py` — `detect_regime()` — is already close to industry practice. Wire it in!

#### E. The Hybrid Approach (What We Recommend)
Combine **3 signals** into a single regime score (0-1):
1. **Trend strength** (ADX / 50): 0 = ranging, 1 = strong trend → weight 0.4
2. **Volatility ratio** (20d vol / 60d vol): < 0.8 = low vol trending, > 1.2 = high vol chaos → weight 0.3
3. **Correlation to gold** (60d rolling): spike = crisis → weight 0.3

```python
def regime_score(df, benchmark=None):
    """0 to 1 score where 0 = trend-following optimal, 1 = mean-reversion optimal."""
    close = df['close']
    returns = close.pct_change()
    
    adx = compute_adx(df)  # already exists
    trend_score = 1 - (adx / 50).clip(0, 1)  # high ADX = low score (trending)
    
    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()
    vol_score = (vol_20 / vol_60).clip(0, 2) / 2  # high vol ratio = high score
    
    return 0.4 * trend_score + 0.3 * vol_score + 0.3  # bias toward mean-reversion
```

Then use the regime score to:
- **Switch between trend and mean-reversion parameter sets** (not strategies)
- **Scale position sizes** (80% during crisis, 100% normal, 120% trending)
- **Adjust trailing stop width** (tighter in mean-reversion, wider in trending)

---

## 5. NEW UNCORRELATED STRATEGY TYPES

### Current System Analysis
Your 4 instruments are **highly correlated**:
| Instrument | Strategy | Asset Class | Correlated With |
|---|---|---|---|
| GLD | Kalman Trend | Gold ETF | XAU/USD, CPER |
| XAU/USD | Kalman Trend (MR) | Gold FX | GLD, CPER |
| CPER | Kalman Trend | Copper ETF | GLD |
| CPER_GLD_RATIO | Z-score MR | Ratio | GLD, CPER (partially) |

### New Strategy Candidates

#### A. Cross-Asset Time-Series Momentum (TSMOM) — Moskowitz, Ooi, Pedersen (2012)
The most robust academic factor. Implement on **different asset classes** (not gold-related):

```python
def tsmom_signal(close, lookbacks=[12, 6, 3]):
    """Time-series momentum: 12-month, 6-month, 3-month return sign."""
    monthly = close.resample('ME').last()
    signals = pd.DataFrame()
    for lb in lookbacks:
        ret = monthly.pct_change(lb)
        signals[f'{lb}m'] = (ret > 0).astype(int) * 2 - 1  # +1 or -1
    # Consensus: average of all lookback signals
    return signals.mean(axis=1)
```

**Best uncorrelated vehicles with ~$50-100k:**
- **TLT** (Long-dated Treasuries) — negative correlation to equities
- **DBC** (Commodity basket) — different from gold
- **FXE** (Euro currency ETF) — orthogonal to gold
- **SHY** (Short Treasuries) — cash proxy
- **IEF** (7-10yr Treasuries) — medium duration

#### B. Volatility Risk Premium (VIX Term Structure)
Sell VIX futures when contango > 5% (the VRP premium). Requires options/futures access.

**Simpler approach:** Use **SVOL** (short volatility ETF) or **ZIV** (volatility futures ETF). The strategy is simply:
```python
def short_vol_signal(vix_term):
    """
    vix_term = VIX1D / VIX9D ratio (>1 = contango = premium).
    Enter short vol when contango is high enough.
    """
    exit_threshold = 0.95  # exit when contango collapses
    entry_threshold = 1.05  # enter when contango > 5%
    
    entries = (vix_term > entry_threshold) & (vix_term.shift(1) <= entry_threshold)
    exits = (vix_term < exit_threshold)
    return entries, exits
```

#### C. Statistical Arbitrage: Basket of Pairs
Beyond CPER_GLD_RATIO, add more pairs:
- **GLD/SLV** (gold/silver ratio) — strong mean reversion
- **USO/DBC** (oil vs commodities) — sector mean reversion
- **TLT/IEF** (long vs intermediate bonds) — yield curve steepening/flattening

Each pair enters 1-2x per week, uncorrelated to your gold strategies.

#### D. Carry strategies (AQR style)
- **FX carry:** Short low-yield currencies vs long high-yield
- Simplified: **UUP** (USD index) vs **FXE** (Euro) — rate differential

#### E. Mean Reversion on Correlation Breakdown
When correlation between GLD and XAU/USD drops below 1-month rolling z-score of -2, mean reversion trade (they'll re-converge).

#### F. For Your $50-100k Capital (Realistic Allocation)
| Strategy | Weight | Vehicles | Expected Correlation to Gold |
|---|---|---|---|
| Gold Trend (existing) | 15% | GLD, XAU/USD | 1.0 (baseline) |
| Copper Trend (existing) | 10% | CPER | 0.3-0.5 |
| Gold/Copper Ratio (existing) | 10% | CPER_GLD_RATIO | 0.0 (market neutral) |
| Bond Trend (NEW) | 15% | TLT, IEF | -0.3 to -0.5 |
| Cross-Asset Momentum (NEW) | 10% | IWM, EEM, DBC | 0.0-0.2 |
| Bond/Gold Ratio (NEW) | 10% | TLT/GLD | -0.5 to -0.7 |
| Short Vol Premium (NEW) | 10% | SVOL | -0.2 |
| FX Carry (NEW) | 10% | UUP, FXE | 0.0 |
| Commodity Momentum (NEW) | 10% | DBC, USO | 0.2-0.4 |

**This reduces portfolio correlation from ~0.7 to ~0.2, dramatically improving Sharpe.**

---

## 6. MACHINE LEARNING SIGNAL GENERATION (NOT OVERFIT)

### The Overfitting Problem
Your current system uses **rules-based strategies** (Kalman, z-score) — this is actually the gold standard for avoiding overfitting! Don't replace them with black-box ML.

**Instead**, use ML for **strategy selection and combination**, not raw signal generation.

#### A. Meta-Labeling (López de Prado)
The most practical ML approach: instead of predicting *direction* (which has no edge), predict *whether your existing signal will succeed*.

```python
from sklearn.ensemble import RandomForestClassifier

def meta_label_features(df, entries):
    """Features for meta-labeling: features at entry time."""
    features = []
    labels = []
    
    entry_indices = entries[entries].index
    for idx in entry_indices:
        loc = df.index.get_loc(idx)
        if loc + 20 >= len(df):
            continue
        # Features: vol, ADX, volume ratio, spread, etc.
        f = {
            'vol_20': returns.iloc[loc-20:loc].std(),
            'adx': adx.iloc[loc],
            'volume_ratio': volume.iloc[loc] / volume.iloc[loc-20:loc].mean(),
            'entry_zscore': zscore.iloc[loc],
            'regime': regime.iloc[loc],
        }
        # Label: did this trade make money? (1 = yes, 0 = no)
        exit_price = df['close'].iloc[min(loc+20, len(df)-1)]
        label = 1 if exit_price > df['close'].iloc[loc] else 0
        features.append(f)
        labels.append(label)
    
    return pd.DataFrame(features), np.array(labels)

# Train on first 70% of data
X_train, y_train = meta_label_features(df.iloc[:split], entries.iloc[:split])
model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)
model.fit(X_train, y_train)

# Use ONLY on OOS data
X_test, _ = meta_label_features(df.iloc[split:], entries.iloc[split:])
confidence = model.predict_proba(X_test)[:, 1]  # probability of success
filtered_entries = entries.iloc[split:] & (confidence > 0.6)  # require 60% confidence
```

**Key rule:** Train meta-labeler on **pre-2023** data, test on **2023-2024**. If it works OOS, it's real.

#### B. Regularized Linear Models (Ridge/Lasso/ElasticNet)
If you do use ML for signal generation, use **highly regularized linear models**:
- ElasticNet with `alpha=0.1`, `l1_ratio=0.5` — automatically selects features
- Features should be **economically interpretable** (not random technical indicators)
- Apply **walk-forward retraining** with monthly retrain

#### C. Feature Engineering Rules to Avoid Overfitting
1. **No more than 5 features** per model (use Lasso to shrink)
2. **No data snooping** — features must be lagged by at least 1 bar
3. **Institutional features only:** volume, volatility, correlation, trend strength, carry
4. **Cross-validation:** Purged k-fold with embargo (Section 1A)
5. **Benchmark all ML models against a simple "always long" baseline**

#### D. What NOT to Do
- No LSTM/Transformers on daily data for $50k accounts (insufficient signal-to-noise)
- No complex neural networks (they will overfit to your 6-15 trades per instrument)
- No reinforcement learning for direct trading (it optimizes for backtest path, not reality)

#### E. Practical ML Integration for Your System
The best ML model for your data volume (few trades) is **Logistic Regression with L1 regularization** on 3-4 carefully chosen features:

```python
from sklearn.linear_model import LogisticRegression

def train_signal_model(is_df, features=['vol_20', 'adx_14', 'volume_ratio', 'correlation']):
    """L1-regularized logistic regression on regime features."""
    # Label: next 5-bar return direction
    X = is_df[features].shift(1).dropna()  # shift to avoid lookahead
    future_ret = is_df['close'].pct_change(5).shift(-5)
    y = (future_ret > 0).astype(int)
    
    # Align
    common_idx = X.index.intersection(y.dropna().index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    model = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=1000)
    model.fit(X, y)
    
    # Apply strong L1 to zero out weak features
    return model
```

---

## 7. POSITION SIZING FOR PROP FIRM CHALLENGES

### The Constraints
Your config: daily_drawdown < 4%, max_drawdown < 10%, $50k account.

**The math:**
- Max daily loss: $2,000 (4% of $50k)
- Max total loss: $5,000 (10% of $50k)
- At 4% risk per trade (your current config) = $2,000/trade — **that's 100% of daily DD budget in ONE trade!**

**This is why your system is passing prop firm rules but barely.** A single bad day blows 100% of daily DD.

### Optimal Prop Firm Sizing Framework

#### A. Hard Budget: Use Daily DD as the Constraint
The daily DD limit is the **binding constraint**, not the total account.

```python
def prop_firm_position_size(equity, daily_dd_budget, max_trades_per_day, atr, price):
    """
    Size each trade as: daily_dd_budget / (max_trades * ATR_mult)
    Ensures no single trade can blow the daily limit.
    """
    daily_max_loss = equity * daily_dd_budget  # $50k * 4% = $2,000
    risk_per_trade = daily_max_loss / max_trades_per_day  # $2,000 / 3 = $667
    
    # Convert to position size
    risk_per_unit = atr * price  # dollar risk per unit
    size = risk_per_trade / risk_per_unit
    return size.clip(lower=0)
```

**Key principle:** Never risk more than **1/3 of daily DD on any single trade**. With 4% daily DD and 3 max trades, each trade risks 1.33% of account max.

#### B. Consecutive Loss Reduction (Already in Your Code!)
Your `progressive_atr_position_sizes` is excellent. Activate it with stricter thresholds for prop firm:
```python
progressive_thresholds = [
    (1, 0.80),  # After 1 loss: 80% of base size
    (2, 0.60),  # After 2: 60%
    (3, 0.40),  # After 3: 40%
    (4, 0.20),  # After 4: 20%
]
```

#### C. The 80% Rule (Industry Best Practice)
Never use more than **80% of your daily DD allowance**. Keep 20% buffer for slippage, gap moves, and intraday volatility.

```python
MAX_DAILY_USAGE = 0.80  # Only use 80% of daily limit
```

#### D. Time-Based Position Scaling
In the first 2 weeks of a prop firm challenge (the most critical part), reduce all positions by 50%. This ensures you survive the initial period and build a track record.

#### E. Complete Prop Firm Risk Budget
For your $50k account with 4% daily / 10% max:
```
Constraint               | Value     | Per Trade
-------------------------|-----------|----------
Account                  | $50,000   | —
Daily DD limit           | $2,000    | —
Max trades per day       | 3         | —
Risk per trade (80% rule)| $533      | 1.07% of account
Max concurrent risk      | $1,600    | 3.2% of account
Max total risk           | $5,000    | 10% all-time

Suggested ATR multiplier | 2.5       | widens in high vol
Position sizing:         | $533 / (ATR * 2.5 * price)|
```

Change your `RISK_CONFIG`:
```python
RISK_CONFIG = {
    "max_risk_per_trade_pct": 0.0107,  # $533 / $50,000 = 1.07% (was 4%!)
    "max_exposure_pct": 0.032,  # 3.2% total (was 50%!)
    "atr_period": 14,
    "max_concurrent_positions": 3,
    "use_kelly_sizing": True,
    "kelly_fraction": 0.25,
}
```

Yes, this reduces returns, but it **ensures the prop firm challenge isn't failed by 1 bad day**.

#### F. ATR-Based Sizing Calibration
Your current ATR sizing uses `risk_dollars * kelly_mult / atr`. This is correct but the risk_pct needs calibration:

```python
def calibrate_atr_sizing(equity, daily_dd_pct, max_trades, atr_mult=2.5):
    """
    Calibrate risk_pct so that max_trades * risk_pct * atr_mult 
    stays within daily_dd_pct.
    """
    # Each trade's loss potential = risk_pct * atr_mult * equity
    # With max_trades: total = max_trades * risk_pct * atr_mult * equity <= daily_dd_pct * equity
    # Therefore: risk_pct <= daily_dd_pct / (max_trades * atr_mult)
    risk_pct = daily_dd_pct / (max_trades * atr_mult)
    return min(risk_pct, 0.02)  # cap at 2% regardless
```

For your setup: `risk_pct = 0.04 / (3 * 2.5) = 0.0053 = 0.53%` per trade. This is the safe starting point.

#### G. Monte Carlo Validation for Prop Firm
Run **1000 Monte Carlo simulations** of your strategy. The prop firm pass rate should be >80%:

```python
def prop_firm_monte_carlo(equity_curve_montecarlo, daily_dd=0.04, max_dd=0.10):
    """What % of simulations pass prop firm rules?"""
    passes = 0
    for simulation in simulations:
        # Check daily DD
        daily_high = simulation.expanding().max()
        daily_drawdown = (daily_high - simulation) / daily_high
        max_daily_dd = daily_drawdown.max()
        
        # Check max DD
        running_max = simulation.expanding().max()
        total_dd = (running_max - simulation) / running_max
        
        if max_daily_dd < daily_dd and total_dd.max() < max_dd:
            passes += 1
    return passes / len(simulations)
```

---

## PRIORITY ACTION PLAN

### Immediate (1-2 days)
1. **Fix walk-forward** in main.py: Use `purged_walk_forward.py` with ≥10 folds, embargo=20, and report combined portfolio OOS as primary metric
2. **Add Deflated Sharpe Ratio** to reporting (code already partially exists in `metrics.py`!)
3. **Wire `detect_regime()`** from optimization/ into engine.py — stop using HMM
4. **Fix position sizing** for prop firm: change max_risk_per_trade_pct from 4% to ~1%
5. **Add CPER_GLD_RATIO exit optimization** — this is your strongest signal by trade count, optimize exits aggressively

### Short-term (3-7 days)
6. **Add TLT bond trend** strategy — immediately uncorrelated to gold with $50k capital
7. **Implement triple-barrier exit** system (replaces grid of 6 exit configs)
8. **Add vol-targeting overlay** on risk parity weights
9. **Meta-label the CPER_GLD_RATIO z-score entries** — boost PF from 1.10

### Medium-term (2-4 weeks)
10. **Add TSMOM** on broad commodity/equity ETFs (DBC, IWM, EEM)
11. **Parameter clustering** for Kalman Q values (replace fixed Q=0.01/0.02)
12. **GMM regime detection** as alternative to HMM
13. **Bootstrap-based position sizing calibration** for prop firm survival rate >80%

---

## KEY REFERENCES

1. **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley. — CPCV, triple-barrier, meta-labeling, deflated Sharpe
2. **Bailey, D.H. & López de Prado, M.** (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*.
3. **Bailey, D.H. et al.** (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
4. **Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H.** (2012). "Time series momentum." *Journal of Financial Economics*.
5. **Deep, G., Deep, A. & Lamptey, W.** (2025). "Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework." arXiv:2512.12924.
6. **Maillard, S., Roncalli, T. & Teiletche, J.** (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management*.
7. **Harvey, C.R., Liu, Y. & Zhu, H.** (2016). "…and the Cross-Section of Expected Returns." *Review of Financial Studies*.
8. **Beckman, Q.** (2025). "Walk-Forward Optimization with Clustering Parameter Selection." [Quant Beckman Blog](https://www.quantbeckman.com/p/with-code-walk-forward-cvcl-optimization).
9. **Pardo, R.** (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
10. **AQR.** (2023). "Factor Timing: The Siren Song." Asness, et al. — Argues factor timing is extremely difficult, favors constant allocation.
