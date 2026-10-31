"""Builds notebooks/SignalForge_verify.ipynb (run: python scripts/build_colab_notebook.py)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# SignalForge: independent verification notebook

This notebook checks SignalForge **without using any SignalForge code**:

1. **Get real NSE data** from Kaggle (NIFTY-50, 2000–2021), Hugging Face (`indian-market-historical-ohlcv`) or yfinance.
2. **Verify the data cleaning**: the Kaggle prices are *not* split-adjusted. We adjust them and check the result
   against Yahoo's split-adjusted prices.
3. **Reproduce the baseline experiment** (logistic regression on RELIANCE) from scratch and compare the numbers with
   the SignalForge dashboard.
4. **Show why chronological validation matters**: the same model scored with a shuffled split.

Run top to bottom in Google Colab (Runtime → Run all). Every step prints what it checked."""),
    code("!pip -q install yfinance pyarrow huggingface_hub kagglehub scikit-learn pandas numpy matplotlib"),
    md("""## 1. Load the data

Pick **one** source with `SOURCE`. Kaggle needs your Kaggle login (kagglehub asks for it), or upload `archive.zip`
yourself with the file picker."""),
    code('''import io, zipfile, glob, os
import numpy as np, pandas as pd

SOURCE = "kaggle_upload"   # "kaggle_upload" | "kagglehub" | "huggingface" | "yfinance"
SYMBOL = "RELIANCE"

def load_kaggle_from_zip(zbytes, symbol):
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    return pd.read_csv(io.BytesIO(z.read(f"{symbol}.csv")), parse_dates=["Date"])

if SOURCE == "kaggle_upload":
    from google.colab import files                      # opens a file picker: choose archive.zip
    up = files.upload()
    raw = load_kaggle_from_zip(next(iter(up.values())), SYMBOL)
elif SOURCE == "kagglehub":
    import kagglehub
    path = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")
    raw = pd.read_csv(os.path.join(path, f"{SYMBOL}.csv"), parse_dates=["Date"])
elif SOURCE == "huggingface":
    raw = pd.read_parquet(f"hf://datasets/vishnun0027/indian-market-historical-ohlcv/stocks/{SYMBOL}.parquet")
    raw = raw.rename(columns=str.title).rename(columns={"Adj_Close": "Adj Close"})
    raw["Date"] = pd.to_datetime(raw["Date"])
else:
    import yfinance as yf
    raw = yf.download(f"{SYMBOL}.NS", start="2000-01-01", end="2021-05-01", auto_adjust=False, progress=False)
    raw.columns = raw.columns.get_level_values(0) if hasattr(raw.columns, "levels") else raw.columns
    raw = raw.reset_index()

raw = raw.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
print(SOURCE, SYMBOL, raw.shape, raw["Date"].min().date(), "→", raw["Date"].max().date())
raw.head()'''),
    md("""## 2. Find and fix splits / bonus issues

An unadjusted 1:1 bonus halves the share price overnight. To a model that looks like a 50% crash.
Rule (same idea SignalForge uses): an overnight drop of more than 33% between consecutive sessions is treated as a
corporate action; the ratio is snapped to a standard split ratio when within 5%, otherwise the overnight gap is used.
All earlier prices are multiplied by that ratio."""),
    code('''CANDIDATES = [1/2, 1/3, 1/4, 1/5, 1/10, 2/3, 2/5]

def adjust_splits(df):
    df = df.copy()
    prev = df["Close"].shift(1)
    gap = df["Open"] / prev
    consecutive = df["Date"].diff().dt.days <= 5
    events, mult = [], np.ones(len(df))
    for i in np.flatnonzero(((gap < 0.67) & consecutive).fillna(False).values):
        best = min(CANDIDATES, key=lambda c: abs(gap.iloc[i] / c - 1))
        f = best if abs(gap.iloc[i] / best - 1) < 0.05 else gap.iloc[i]
        events.append((df["Date"].iloc[i].date(), round(float(f), 4)))
        mult[:i] *= f
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c] * mult
    df["Volume"] = df["Volume"] / mult   # a 1:2 split doubles the share count, so earlier volume doubles too
    return df, events

before = raw["Close"].pct_change().abs().gt(0.4).sum()
adj, events = adjust_splits(raw)
after = adj["Close"].pct_change().abs().gt(0.4).sum()
print("Corporate actions found:", events)
print(f"Daily moves larger than 40%: {before} before adjustment -> {after} after")'''),
    md("""### Cross-check against Yahoo's split-adjusted prices

If our adjustment is right, our daily returns should match Yahoo's (which are split-adjusted by Yahoo) almost exactly."""),
    code('''import yfinance as yf
y = yf.download(f"{SYMBOL}.NS", start=str(adj["Date"].min().date()), end=str(adj["Date"].max().date()),
                auto_adjust=False, progress=False)
y.columns = y.columns.get_level_values(0) if hasattr(y.columns, "levels") else y.columns
m = pd.DataFrame({"ours": adj.set_index("Date")["Close"].pct_change(),
                  "yahoo": y["Close"].pct_change()}).dropna()
corr = m["ours"].corr(m["yahoo"])
bad = (m["ours"] - m["yahoo"]).abs().gt(0.02)
print(f"Overlapping days: {len(m)}  correlation of daily returns: {corr:.4f}  days differing by >2 pts: {bad.sum()}")
m[bad].head(10)'''),
    md("""## 3. Reproduce the SignalForge baseline from scratch

Same settings as the **Baseline: Logistic Regression** preset:
8 features, target = next-day close-to-close return > 0, first 70% of usable rows for training, last training row
purged (its label looks into the test period), StandardScaler + LogisticRegression(C=0.1) fitted on training rows
only, long when P(up) ≥ 0.5, position decided at yesterday's close, 10 bps commission + 5 bps slippage on every
position change."""),
    code('''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

d = adj.rename(columns=str.lower)
c = d["close"]
def rsi(s, n=14):
    delta = s.diff(); g = delta.clip(lower=0).rolling(n).mean(); l = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
X = pd.DataFrame({
    "ret_1": c.pct_change(1), "ret_5": c.pct_change(5), "ret_20": c.pct_change(20),
    "sma_ratio_10": c / c.rolling(10).mean() - 1, "sma_ratio_50": c / c.rolling(50).mean() - 1,
    "rsi_14": rsi(c), "vol_20": c.pct_change().rolling(20).std(),
    "volume_z_20": (d["volume"] - d["volume"].rolling(20).mean()) / d["volume"].rolling(20).std(),
}).replace([np.inf, -np.inf], np.nan)
fwd = c.shift(-1) / c - 1
y_all = (fwd > 0).astype(float).where(fwd.notna())
frame = X.assign(y=y_all, close=c, date=d["date"]).dropna().set_index("date")

cut = int(len(frame) * 0.7)
train, test = frame.iloc[:cut].iloc[:-1], frame.iloc[cut:]       # purge 1 row (horizon = 1)
feats = list(X.columns)
model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=42))])
model.fit(train[feats], train["y"])
p = model.predict_proba(test[feats])[:, 1]

acc = accuracy_score(test["y"], (p >= 0.5).astype(int))
base = max(test["y"].mean(), 1 - test["y"].mean())
ret = test["close"].pct_change().fillna(0)
pos = pd.Series((p >= 0.5).astype(float), index=test.index).shift(1).fillna(0)
turn = pos.diff().abs(); turn.iloc[0] = pos.iloc[0]
net = pos * ret - turn * 0.0015
r = net.iloc[1:]
sharpe = r.mean() / r.std(ddof=0) * np.sqrt(252)
eq = (1 + r).cumprod()
bh = ret.iloc[1:]; bh_sharpe = bh.mean() / bh.std(ddof=0) * np.sqrt(252)
print(f"Train {train.index[0].date()} → {train.index[-1].date()} ({len(train)} rows) | Test {test.index[0].date()} → {test.index[-1].date()} ({len(test)} rows)")
print(f"Test accuracy {acc:.1%} (majority baseline {base:.1%}) | AUC {roc_auc_score(test['y'], p):.3f}")
print(f"Test return after costs {eq.iloc[-1]-1:.1%} | Sharpe {sharpe:.2f} | max drawdown {(eq/eq.cummax()-1).min():.1%} | buy-and-hold Sharpe {bh_sharpe:.2f}")
print("Compare with the SignalForge experiment page for the same stock and preset.")'''),
    md("""## 4. Why the split must be chronological

Same model, same features, but rows shuffled before splitting. Shuffling lets the model train on days *after* the
days it is tested on. Compare the two accuracies and the two Sharpe-like scores."""),
    code('''from sklearn.model_selection import train_test_split
tr_s, te_s = train_test_split(frame, test_size=0.3, shuffle=True, random_state=0)
m2 = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=2000))]).fit(tr_s[feats], tr_s["y"])
print(f"Chronological test accuracy: {acc:.1%}")
print(f"Shuffled test accuracy:      {accuracy_score(te_s['y'], m2.predict(te_s[feats])):.1%}")
print("With a leaky feature the gap becomes dramatic; try adding a centered moving average below.")
sma_centered = pd.Series((c / c.rolling(11, center=True).mean() - 1).values, index=d["date"])  # peeks 5 days ahead
fl = frame.assign(sma_centered=sma_centered.reindex(frame.index)).dropna()
feats2 = feats + ["sma_centered"]
cut2 = int(len(fl) * 0.7)
m3 = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=2000))]).fit(fl.iloc[:cut2-1][feats2], fl.iloc[:cut2-1]["y"])
print(f"Chronological accuracy WITH a look-ahead feature: {accuracy_score(fl.iloc[cut2:]['y'], m3.predict(fl.iloc[cut2:][feats2])):.1%}  <- too good to be true")'''),
    md("""## What this notebook proves

* The Kaggle data needs split adjustment, and after adjustment it matches Yahoo's split-adjusted returns.
* The SignalForge baseline numbers can be reproduced with ~40 lines of plain pandas/scikit-learn.
* Shuffled splits and look-ahead features inflate accuracy; SignalForge blocks both."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).resolve().parent.parent / "notebooks" / "SignalForge_verify.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
