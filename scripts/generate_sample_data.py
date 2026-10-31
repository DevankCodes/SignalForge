"""Generate synthetic daily OHLCV data for demos (clearly labelled DEMO_*; not real market data).

Returns follow a regime-switching process (calm bull / volatile bear) with a weak short-term
momentum effect, so honest models find a small, unstable edge, which is what real markets look like.
Also writes two intentionally broken files to sample_data/bad/ to demo validation.

Usage:  python scripts/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"

ASSETS = {
    # name: (seed, start_price, vol_scale, drift_scale, momentum)
    "DEMO_TECH": (11, 420.0, 1.35, 1.3, 0.15),
    "DEMO_BANK": (22, 1150.0, 1.10, 0.9, 0.11),
    "DEMO_PHARMA": (33, 760.0, 0.95, 0.8, 0.09),
    "DEMO_INDEX": (44, 9800.0, 0.70, 0.9, 0.13),
}


def simulate(seed, p0, vol_scale, drift_scale, phi, start="2014-01-01", end="2025-12-31"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    # Markov regimes: 0 = bull (calm), 1 = bear (volatile)
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.995 if regime[t - 1] == 0 else 0.985
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    mu = np.where(regime == 0, 0.0006, -0.0007) * drift_scale
    sig = np.where(regime == 0, 0.010, 0.021) * vol_scale
    momentum = np.where(regime == 0, phi, phi * 1.6)
    eps = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mu[t] + momentum[t] * r[t - 1] + sig[t] * eps[t]
    close = p0 * np.exp(np.cumsum(r))
    intraday = np.abs(rng.normal(0, sig * 0.8, n))
    open_ = close * np.exp(rng.normal(0, sig * 0.35, n))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    base_vol = 2_000_000 / np.sqrt(p0 / 100)
    volume = base_vol * (1 + 25 * np.abs(r)) * np.exp(rng.normal(0, 0.25, n)) * np.where(regime == 1, 1.4, 1.0)
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"), "Open": open_.round(2), "High": high.round(2),
        "Low": low.round(2), "Close": close.round(2), "Volume": volume.round(0).astype(int),
    })


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "bad").mkdir(exist_ok=True)
    for name, (seed, p0, vs, ds, phi) in ASSETS.items():
        simulate(seed, p0, vs, ds, phi).to_csv(OUT / f"{name}.csv", index=False)
        print("wrote", name)

    # Messy-but-usable file: every common data problem, fixed or flagged by the validator.
    messy = simulate(55, 300.0, 1.0, 1.0, 0.05, start="2016-01-01", end="2025-12-31")
    messy = messy.rename(columns={"Date": "timestamp", "Close": "close price", "Volume": "vol"})
    messy["close price"] = messy["close price"].astype(object)
    messy.loc[100:103, "close price"] = "N/A"            # non-numeric
    messy.loc[200, "close price"] = -5                    # negative price
    messy.loc[300, "timestamp"] = "not a date"            # bad timestamp
    messy.loc[400:402, "timestamp"] = None                # missing timestamps
    messy = pd.concat([messy, messy.iloc[500:505]])       # duplicates
    messy = pd.concat([messy.iloc[:900], messy.iloc[1100:1200], messy.iloc[900:1100], messy.iloc[1200:]])  # out of order
    messy.loc[messy.index[50], "High"] = 1.0              # high < low
    messy.to_csv(OUT / "bad" / "MESSY_but_usable.csv", index=False)

    # Unusable file: no close column.
    broken = simulate(66, 100.0, 1.0, 1.0, 0.05).drop(columns=["Close"])
    broken.to_csv(OUT / "bad" / "BROKEN_no_close_column.csv", index=False)
    (OUT / "bad" / "BROKEN_too_short.csv").write_text(simulate(77, 100.0, 1.0, 1.0, 0.05).head(120).to_csv(index=False))
    print("wrote bad/ examples")


if __name__ == "__main__":
    main()
