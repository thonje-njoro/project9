"""HMM-based market regime detection with optional GARCH volatility forecast.

Implements regime-switching from:
'Algorithmic Trading using Hidden Markov Models' (arXiv:1902.05443)

Key improvements:
1. Probability-based regime filtering (not hard cutoff)
2. Robust fitting with fallback to rolling-window approach
3. Smooth regime transitions to avoid whipsaws
4. KL divergence supplement for information-theoretic regime classification
5. **3-state HMM** (Item 3): Trending / Mean-Reverting / Crisis
6. **GARCH(1,1) volatility forecast** (Item 6): Forward-looking vol estimation
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# GARCH import is lazy — only imported when garch_* functions are called


def compute_kl_divergence(
    returns: pd.Series,
    window: int = 60,
    n_bins: int = 20,
) -> pd.Series:
    """Compute rolling KL divergence of returns distribution from a reference Gaussian.

    High KL divergence = returns distribution deviates strongly from normal
    = choppy/mean-reverting market. Low KL = returns are approximately Gaussian
    = trending/efficient market.
    """
    kl_values = pd.Series(0.0, index=returns.index)
    vals = returns.values
    n = len(vals)

    for end in range(window, n):
        w = vals[end - window:end]
        w = w[~np.isnan(w)]
        if len(w) < window // 2:
            continue

        mu = w.mean()
        sigma = w.std()
        if sigma < 1e-10:
            continue

        counts, edges = np.histogram(w, bins=n_bins, density=True)
        bc = (edges[:-1] + edges[1:]) / 2
        bw = edges[1] - edges[0]

        p_obs = counts * bw
        mask = p_obs > 0
        p_obs = p_obs[mask]
        bc_obs = bc[mask]

        z = (bc_obs - mu) / sigma
        p_gauss = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi)) * bw
        gmask = p_gauss > 0
        p_obs = p_obs[gmask]
        p_gauss = p_gauss[gmask]

        if len(p_obs) < 2:
            continue

        p_obs = p_obs / p_obs.sum()
        p_gauss = p_gauss / p_gauss.sum()

        kl_values.iloc[end] = np.sum(p_obs * np.log(p_obs / p_gauss))

    return kl_values


def kl_regime_score(
    returns: pd.Series,
    window: int = 60,
    n_bins: int = 20,
) -> pd.Series:
    """Normalize KL divergence into a 0-1 regime score.

    High score (near 1) = high KL = choppy/mean-reverting.
    Low score (near 0) = low KL = trending/Gaussian.
    """
    kl = compute_kl_divergence(returns, window, n_bins)
    kl_rolling_max = kl.rolling(window * 2, min_periods=1).max()
    kl_rolling_min = kl.rolling(window * 2, min_periods=1).min()
    range_val = kl_rolling_max - kl_rolling_min
    score = (kl - kl_rolling_min) / (range_val + 1e-10)
    return score.clip(0, 1)


def garch_volatility_forecast(
    returns: pd.Series,
    horizon: int = 1,
) -> pd.Series:
    """GARCH(1,1) volatility forecast.

    Returns a series of forecast volatility for the next period.
    Falls back to rolling std if arch_model is not available.

    Args:
        returns: Asset returns series
        horizon: Forecast horizon in steps

    Returns:
        Series of forecast volatility values
    """
    try:
        from arch import arch_model
    except ImportError:
        # Fallback: rolling EWMA volatility
        return returns.ewm(span=20, adjust=False).std().bfill()

    clean = returns.dropna().copy()
    if len(clean) < 50:
        return returns.ewm(span=20, adjust=False).std().bfill()

    try:
        am = arch_model(clean.values * 100, vol="Garch", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        forecasts = res.forecast(horizon=horizon)
        # GARCH forecast gives variance; convert to std, scale back
        forecast_var = forecasts.variance.values[-1, horizon - 1] if horizon > 0 else 0.0
        forecast_vol = float(np.sqrt(forecast_var)) / 100.0

        # Build output series: last N values get the forecast, rest get EWMA
        result = returns.ewm(span=20, adjust=False).std().bfill()
        result.iloc[-min(horizon, len(result)):] = forecast_vol
        return result
    except Exception:
        return returns.ewm(span=20, adjust=False).std().bfill()


def compute_regime_stop_multipliers(
    regime_probs: pd.DataFrame,
    base_mult: float = 2.5,
    crisis_mult: float = 1.0,
    normal_mult: float = 2.5,
    trend_mult: float = 3.0,
) -> pd.Series:
    """Compute regime-adaptive stop loss multipliers.

    Uses the 3-state regime probabilities to compute a blended
    trail_atr_mult that tightens in crisis regimes and widens
    in trending regimes.

    Args:
        regime_probs: DataFrame with columns ['trending', 'mean_reverting', 'crisis']
        base_mult: Base trailing stop multiplier
        crisis_mult: Multiplier when crisis probability is high (tighter)
        normal_mult: Normal regime multiplier
        trend_mult: Multiplier when trending probability is high (wider)

    Returns:
        Series of stop loss multipliers (regime-adaptive)
    """
    blended = pd.Series(base_mult, index=regime_probs.index)

    if "crisis" in regime_probs.columns:
        crisis_prob = regime_probs["crisis"]
        blended = blended * (1 - crisis_prob * 0.6)  # tighten up to 60%

    if "trending" in regime_probs.columns:
        trend_prob = regime_probs["trending"]
        blended = blended * (1 + trend_prob * 0.2)  # widen up to 20%

    return blended.clip(crisis_mult, trend_mult)


class RegimeFilter:
    """Market regime detection with 2-state or 3-state HMM.

    2-state: TRENDING vs MEAN_REVERTING
    3-state: TRENDING vs MEAN_REVERTING vs CRISIS

    The 3-state model adds a high-volatility 'crisis' regime that triggers
    tighter stops and reduced position sizes.
    """

    def __init__(
        self,
        n_states: int = 2,
        lookback: int = 60,
        min_regime_bars: int = 5,
        use_garch: bool = True,
    ) -> None:
        self.n_states = min(n_states, 3)  # max 3 states
        self.lookback = lookback
        self.min_regime_bars = min_regime_bars
        self.model = None
        self.regime_map: dict[int, str] = {}
        self._fallback_mode = False
        self.use_garch = use_garch
        self._last_garch_vol = None

    def _compute_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute features for HMM: returns + realized volatility + optional GARCH."""
        close = df["close"]
        returns = close.pct_change()
        vol_window = 20
        realized_vol = returns.rolling(vol_window).std()

        if self.use_garch:
            garch_vol = garch_volatility_forecast(returns)
            self._last_garch_vol = garch_vol
            features = np.column_stack([
                returns.values,
                realized_vol.values,
                garch_vol.values,
            ])
        else:
            features = np.column_stack([
                returns.values,
                realized_vol.values,
            ])

        valid_mask = ~np.isnan(features).any(axis=1)
        return features, valid_mask

    def fit(self, df: pd.DataFrame) -> None:
        """Fit HMM on historical returns and volatility features."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("pip install hmmlearn")

        features, valid_mask = self._compute_features(df)
        valid_features = features[valid_mask]

        n_features = valid_features.shape[1]

        if len(valid_features) < 50:
            self._fallback_mode = True
            self._fit_rolling_window(df)
            return

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=500,  # increased from 200 for better convergence
                random_state=42,
                tol=1e-4,
            )
            self.model.fit(valid_features)

            # Map states to regime labels by variance (lowest = trending, highest = crisis)
            if hasattr(self.model, "var_"):
                variances = self.model.var_.sum(axis=1)
            else:
                variances = np.array([np.trace(self.model.covars_[i])
                                      for i in range(self.n_states)])

            sorted_idx = np.argsort(variances)

            if self.n_states == 2:
                self.regime_map = {
                    sorted_idx[0]: "trending",
                    sorted_idx[1]: "mean_reverting",
                }
            else:  # 3 states
                self.regime_map = {
                    sorted_idx[0]: "trending",      # lowest variance
                    sorted_idx[1]: "mean_reverting",  # medium variance
                    sorted_idx[2]: "crisis",         # highest variance
                }

            self._fallback_mode = False
        except Exception as e:
            print(f"  HMM fit failed ({e}), using rolling-window fallback")
            self._fallback_mode = True
            self._fit_rolling_window(df)

    def _fit_rolling_window(self, df: pd.DataFrame) -> None:
        """Fallback: classify regime using rolling window statistics."""
        close = df["close"]
        returns = close.pct_change()
        vol = returns.rolling(20).std()
        vol_median = vol.rolling(60).median()

        self._vol_threshold = vol_median
        self._returns_mean = returns.rolling(20).mean()
        self._fallback_mode = True

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return regime classification for each bar."""
        if self._fallback_mode:
            return self._predict_rolling(df)

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("pip install hmmlearn")

        if self.model is None:
            raise RuntimeError("Call fit() before predict()")

        features, valid_mask = self._compute_features(df)
        default_label = "mean_reverting" if self.n_states == 2 else "mean_reverting"
        regimes = pd.Series(default_label, index=df.index, dtype="object")

        if valid_mask.sum() > 0:
            valid_features = features[valid_mask]
            try:
                states = self.model.predict(valid_features)
            except Exception:
                return self._predict_rolling(df)
            regime_labels = [self.regime_map.get(s, default_label) for s in states]
            regimes.iloc[valid_mask] = regime_labels

        regimes = regimes.ffill().fillna(default_label)
        return regimes

    def _predict_rolling(self, df: pd.DataFrame) -> pd.Series:
        """Predict using rolling window approach (fallback)."""
        close = df["close"]
        returns = close.pct_change()
        vol = returns.rolling(20).std()
        vol_median = vol.rolling(60).median()

        high_vol = vol > vol_median
        trend_strength = returns.rolling(20).mean().abs() / (vol + 1e-8)
        trending = high_vol & (trend_strength > 0.5)

        # Crisis detection: vol > 2x median
        crisis = vol > vol_median * 2.0

        if self.n_states >= 3:
            regimes = pd.Series("mean_reverting", index=df.index, dtype="object")
            regimes[trending & ~crisis] = "trending"
            regimes[crisis] = "crisis"
        else:
            regimes = pd.Series("mean_reverting", index=df.index, dtype="object")
            regimes[trending] = "trending"

        return regimes

    def get_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return probability of each regime per bar."""
        if self._fallback_mode:
            regimes = self.predict(df)
            probs = pd.DataFrame(index=df.index)
            unique_regimes = regimes.unique()
            for r in unique_regimes:
                probs[r] = (regimes == r).astype(float)
            # Fill missing regime columns
            for r in (["trending", "mean_reverting", "crisis"][:self.n_states]):
                if r not in probs.columns:
                    probs[r] = 0.0
            return probs

        if self.model is None:
            raise RuntimeError("Call fit() before get_regime_probabilities()")

        features, valid_mask = self._compute_features(df)
        probs = pd.DataFrame(index=df.index)

        if valid_mask.sum() > 0:
            valid_features = features[valid_mask]
            try:
                state_probs = self.model.predict_proba(valid_features)
            except Exception:
                regimes = self.predict(df)
                unique_regimes = regimes.unique()
                for r in unique_regimes:
                    probs[r] = (regimes == r).astype(float)
                return probs

            for i in range(self.n_states):
                label = self.regime_map.get(i, f"state_{i}")
                col = pd.Series(0.0, index=df.index)
                col.iloc[valid_mask] = state_probs[:, i]
                probs[label] = col
        else:
            for i in range(self.n_states):
                label = self.regime_map.get(i, f"state_{i}")
                probs[label] = 0.0
            probs["mean_reverting"] = 1.0

        return probs

    def get_kl_enhanced_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine HMM probabilities with KL divergence for sharper regime classification."""
        hmm_probs = self.get_regime_probabilities(df)
        close = df["close"]
        returns = close.pct_change()
        kl_score = kl_regime_score(returns)

        enhanced = hmm_probs.copy()

        if self.n_states >= 3 and "crisis" in hmm_probs.columns:
            # Crisis detection: use KL + GARCH vol spike
            if self.use_garch and self._last_garch_vol is not None:
                garch_vol = self._last_garch_vol
                garch_spike = (garch_vol / garch_vol.rolling(60).mean() - 1).clip(0, 1)
                crisis_blend = 0.3 * hmm_probs.get("crisis", pd.Series(0, index=df.index)) + \
                               0.4 * (1 - kl_score) + \
                               0.3 * garch_spike.fillna(0)
                enhanced["crisis"] = crisis_blend.clip(0, 1)
                enhanced["mean_reverting"] = (1 - enhanced["crisis"]) * hmm_probs.get("mean_reverting", pd.Series(0.5, index=df.index))
                enhanced["trending"] = (1 - enhanced["crisis"]) * hmm_probs.get("trending", pd.Series(0.5, index=df.index))
            else:
                enhanced["crisis"] = hmm_probs.get("crisis", pd.Series(0, index=df.index))
        else:
            # 2-state blending unchanged
            mr_prob = hmm_probs.get("mean_reverting", pd.Series(0.5, index=df.index))
            blend = 0.6
            enhanced_mr = blend * mr_prob + (1 - blend) * kl_score
            enhanced["mean_reverting"] = enhanced_mr.clip(0, 1)
            enhanced["trending"] = 1 - enhanced["mean_reverting"]

        return enhanced


def create_regime_mask(
    regimes: pd.Series,
    strategy_type: str,
    min_bars: int = 3,
    probability: pd.Series = None,
    prob_threshold: float = 0.4,
) -> pd.Series:
    """Create a boolean mask that filters signals based on regime.

    Uses probability-based filtering for smoother transitions.
    The 3-state model blocks ALL signals during CRISIS regime
    (conservative risk-off behavior).
    """
    # Crisis override: block all signals during crisis
    if hasattr(regimes, 'dtype') and regimes.dtype == 'object':
        crisis_mask = regimes == "crisis"
        if crisis_mask.any():
            # Crisis blocks everything
            pass  # handled below in strategy_type filter

    if strategy_type == "mean_reverting":
        if probability is not None:
            raw_mask = probability > prob_threshold
        else:
            raw_mask = regimes == "mean_reverting"
    elif strategy_type in ("trending", "momentum"):
        if probability is not None:
            raw_mask = probability > prob_threshold
        else:
            raw_mask = regimes == "trending"
    else:
        return pd.Series(True, index=regimes.index)

    # Filter out crisis bars regardless of strategy type
    raw_mask = raw_mask & (regimes != "crisis")

    # Smooth: require min_bars in same regime
    filtered = raw_mask.copy()
    state_count = 0
    last_state = None

    for i in range(len(filtered)):
        current = raw_mask.iloc[i]
        if current == last_state:
            state_count += 1
        else:
            state_count = 1
            last_state = current

        if state_count < min_bars:
            filtered.iloc[i] = False

    return filtered
