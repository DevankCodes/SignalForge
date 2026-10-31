"""Feature engineering with a built-in look-ahead guard.

Every honest feature at row t uses only data from rows <= t (rolling windows, shift(+k)).
The two *_LEAK features deliberately peek into the future so the leakage guard can be demonstrated.

`detect_lookahead` proves a feature is causal with a truncation test: compute the feature on the
full history and again on history cut off at day k. A causal feature gives the same value on day k
either way. If the value changes once later data is added, the feature used the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# name -> (group, description, function(df) -> Series)
FEATURES: dict[str, tuple[str, str, callable]] = {
    "ret_1": ("Returns", "1-day return", lambda d: d["close"].pct_change(1)),
    "ret_5": ("Returns", "5-day return", lambda d: d["close"].pct_change(5)),
    "ret_10": ("Returns", "10-day return", lambda d: d["close"].pct_change(10)),
    "ret_20": ("Returns", "20-day return", lambda d: d["close"].pct_change(20)),
    "sma_ratio_10": ("Trend", "Close vs 10-day moving average", lambda d: d["close"] / d["close"].rolling(10).mean() - 1),
    "sma_ratio_50": ("Trend", "Close vs 50-day moving average", lambda d: d["close"] / d["close"].rolling(50).mean() - 1),
    "sma_cross_10_50": ("Trend", "10-day MA vs 50-day MA", lambda d: d["close"].rolling(10).mean() / d["close"].rolling(50).mean() - 1),
    "momentum_60": ("Momentum", "60-day momentum", lambda d: d["close"] / d["close"].shift(60) - 1),
    "rsi_14": ("Momentum", "14-day RSI", lambda d: _rsi(d["close"], 14)),
    "dist_high_20": ("Momentum", "Distance from 20-day high", lambda d: d["close"] / d["close"].rolling(20).max() - 1),
    "vol_10": ("Volatility", "10-day volatility", lambda d: d["close"].pct_change().rolling(10).std()),
    "vol_20": ("Volatility", "20-day volatility", lambda d: d["close"].pct_change().rolling(20).std()),
    "hl_range": ("Volatility", "High-low range / close", lambda d: (d["high"] - d["low"]) / d["close"]),
    "volume_z_20": ("Volume", "Volume z-score (20 days)", lambda d: (d["volume"] - d["volume"].rolling(20).mean()) / d["volume"].rolling(20).std()),
    "volume_ratio_5_20": ("Volume", "5-day vs 20-day average volume", lambda d: d["volume"].rolling(5).mean() / d["volume"].rolling(20).mean() - 1),
}
# Many lagged returns: lets a flexible model memorise noise (used by the Overfit Trap preset).
for _k in range(2, 11):
    FEATURES[f"lag_ret_{_k}"] = (
        "Lagged returns", f"1-day return {_k} days ago",
        (lambda k: (lambda d: d["close"].pct_change().shift(k)))(_k),
    )

# Deliberately leaky features (blocked unless the user explicitly allows look-ahead).
LEAKY_FEATURES: dict[str, tuple[str, str, callable]] = {
    "sma_centered_LEAK": (
        "Leaky (demo)", "Centered 11-day average: uses 5 FUTURE days (common mistake)",
        lambda d: d["close"] / d["close"].rolling(11, center=True).mean() - 1,
    ),
    "next_day_return_LEAK": (
        "Leaky (demo)", "Tomorrow's return: pure look-ahead",
        lambda d: d["close"].shift(-1) / d["close"] - 1,
    ),
}

ALL_FEATURES = {**FEATURES, **LEAKY_FEATURES}
DEFAULT_FEATURES = ["ret_1", "ret_5", "ret_20", "sma_ratio_10", "sma_ratio_50", "rsi_14", "vol_20", "volume_z_20"]
VOLUME_FEATURES = {"volume_z_20", "volume_ratio_5_20"}


def feature_catalog() -> list[dict]:
    out = []
    for name, (group, desc, _) in ALL_FEATURES.items():
        out.append({"name": name, "group": group, "description": desc, "leaky": name in LEAKY_FEATURES})
    return out


def compute_features(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for n in names:
        if n not in ALL_FEATURES:
            raise ValueError(f"Unknown feature '{n}'")
        out[n] = ALL_FEATURES[n][2](df)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_target(df: pd.DataFrame, horizon: int, threshold: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Label = 1 if the forward `horizon`-day return exceeds `threshold`.

    The forward return is the ONLY thing allowed to look ahead, because it is what we predict.
    Rows whose forward window runs past the data are dropped (NaN target).
    """
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    label = (fwd > threshold).astype(float)
    label[fwd.isna()] = np.nan
    return label, fwd


def detect_lookahead(df: pd.DataFrame, names: list[str], n_checks: int = 6, seed: int = 0) -> dict:
    """Truncation test for every feature. Returns {feature: {"leaky": bool, "checked_at": [...]}}."""
    rng = np.random.default_rng(seed)
    n = len(df)
    lo, hi = max(220, int(n * 0.3)), n - 15
    if hi <= lo:
        lo, hi = n // 2, n - 2
    cuts = sorted(set(rng.integers(lo, hi, size=n_checks).tolist()))
    full = compute_features(df, names)
    result = {}
    for name in names:
        leaky = False
        for k in cuts:
            trunc_val = compute_features(df.iloc[:k], [name])[name].iloc[-1]
            full_val = full[name].iloc[k - 1]
            both_nan = pd.isna(trunc_val) and pd.isna(full_val)
            if not both_nan and (pd.isna(trunc_val) != pd.isna(full_val) or not np.isclose(trunc_val, full_val, rtol=1e-9, atol=1e-12)):
                leaky = True
                break
        result[name] = {"leaky": leaky, "checked_at": [df["date"].iloc[k - 1].strftime("%Y-%m-%d") for k in cuts]}
    return result
