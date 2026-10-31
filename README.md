# SignalForge

A quantitative research and experiment management terminal. Upload historical market data, configure
experiments, evaluate predictive signals, and compare hypothetical trading performance, with guards
against the mistakes that make backtests lie: data leakage, look-ahead bias, overfitting and ignoring costs.

**Stack:** FastAPI + SQLite backend, scikit-learn models, vanilla JS + Chart.js frontend served by the same
process. One command to run, one container to deploy.

---

## Run locally (2 minutes)

```bash
cd signalforge
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://localhost:8000. On first start it creates a demo project with 4 synthetic assets, a messy
file and 6 preset experiments (they finish in about 20 seconds).

To start over: stop the server and delete the `data/` folder.

### Real NSE data

Upload any of these on the Datasets page, exactly as downloaded:

- **Kaggle NIFTY-50 (2000–2021)**: `archive.zip` as-is. Every stock becomes a dataset; `stock_metadata.csv` and
  the duplicate `NIFTY50_all.csv` are skipped. These prices are **not** split-adjusted, so SignalForge reads
  NSE's *Prev Close* column (which the exchange adjusts on the ex-date) and back-adjusts earlier prices for
  every split and bonus issue. Renamed tickers (e.g. INFOSYSTCH → INFY) are merged into one series.
- **Hugging Face `vishnun0027/indian-market-historical-ohlcv`**: any `stocks/*.parquet` or `indices/*.parquet`
  file (one or several zipped together). Yahoo suffixes like `.NS` are stripped from names.
- **yfinance**: `pip install yfinance && python scripts/fetch_yfinance.py RELIANCE.NS TCS.NS ^NSEI`, then upload
  the CSVs from `real_data/`.

**Bundle real data into the demo:** put files (CSV, Parquet or ZIP) in `sample_data/real/`. On first start
SignalForge creates an "NSE stocks (real data)" project from them and runs all six presets on RELIANCE (or the
first stock found), testing generalization on the others.

---

## Verify it (tests + Google Colab)

- **Tests:** `pip install -r requirements-dev.txt && pytest -q` runs 19 checks in about 12 seconds: trade timing,
  costs, Sharpe/drawdown/CAGR against hand calculations, purging, train-only scaling, the look-ahead test,
  split adjustment (including RELIANCE's real 2017 bonus) and reproducibility.
- **Colab:** open https://colab.research.google.com → File → Upload notebook → `notebooks/SignalForge_verify.ipynb`
  → Runtime → Run all. When the file picker appears, choose the Kaggle `archive.zip`. It shares no code with the app:
  it adjusts splits itself, cross-checks the adjusted returns against Yahoo Finance, and rebuilds the RELIANCE
  baseline. Expected: test accuracy 50.7%, net return 95.2%, Sharpe 0.53, the same as the app. Set `SOURCE` to
  `"huggingface"` or `"yfinance"` to load data from those instead.

## Model comparison (random forest vs XGBoost and others)

`pip install xgboost && python scripts/model_comparison.py RELIANCE` runs logistic regression, random forest,
sklearn gradient boosting and XGBoost through the same engine (same features, split, purge, costs) and writes
`scripts/model_comparison_RELIANCE.csv`. On RELIANCE every regularized model lands at 50.7–51.7% test accuracy,
inside the ±2.5-point margin and below the 52.0% always-guess baseline; library-default forests and XGBoost
overfit (98–100% train, ~49% test).

## Deploy (Render, free, about 5 minutes)

1. Push this folder to a new GitHub repo:
   ```bash
   git init && git add . && git commit -m "SignalForge"
   git branch -M main
   git remote add origin https://github.com/<you>/signalforge.git
   git push -u origin main
   ```
2. Go to https://dashboard.render.com → **New** → **Blueprint** → pick the repo. Render reads `render.yaml`
   and builds the Dockerfile. (Or **New → Web Service → Docker**, same result.)
3. Wait for the build (3–5 min). Your link is `https://signalforge-xxxx.onrender.com`.

Notes:
- The free tier sleeps after 15 minutes idle; the first visit then takes about a minute. **Open the link
  a few minutes before judging.**
- The free disk is temporary, so the demo workspace is re-created on every restart. That is intentional:
  the deployed link never opens empty.

**Railway alternative:** New Project → Deploy from GitHub repo. It detects the Dockerfile and sets `$PORT`.

**Docker anywhere:** `docker build -t signalforge . && docker run -p 8000:8000 signalforge`

---

## How the problem statement is covered

| Requirement | Where it lives |
|---|---|
| Research workspace: projects and experiments | Projects page, project selector, experiment list (`backend/db.py`) |
| Dataset management: CSV upload + validation of columns, timestamps, missing values | `backend/data.py::validate_csv`, Datasets page |
| Data preview: summaries, missing-value stats, price and volume charts | Dataset detail panel |
| Feature engineering: returns, moving averages, momentum, volatility | `backend/features.py` (24 causal features + 2 deliberately leaky demo features) |
| Experiment configuration: features, model settings, target definition, evaluation periods | New experiment page |
| Model evaluation vs a simple benchmark | Majority-class predictor (accuracy) and buy-and-hold (strategy) on every run |
| Chronological validation | Time-ordered split + **purging** of training rows whose labels overlap the test period + optional embargo |
| Backtesting with configurable transaction costs | `engine.backtest`: commission + slippage in bps charged on every position change |
| Dashboard: predictive metrics, cumulative returns, drawdown | Experiment page: three panels answer three different questions |
| Experiment history | Every run stores config, dataset checksum, seed, results; versions with config diffs |
| Predictions vs actual outcomes | Last test days with P(up), prediction and what actually happened, plus the latest signal after the data ends |
| **Advanced:** LR, random forest, gradient boosting (+ decision tree) | `engine.MODELS` |
| Rolling-window evaluation | Walk-forward with expanding or rolling window, per-fold table and chart |
| Generalization to unseen assets | Trained model applied unchanged to other datasets |
| Market regimes | Bull/bear (200-day MA) and high/low volatility, labelled with past data only |
| Allocation constraints + combining signals into a portfolio | Compare page → Combine into a portfolio: max weight per stock, max total exposure, costs on weight changes, vs equal-weight buy-and-hold (`engine.build_portfolio`) |
| Background jobs + live progress | Thread pool + Server-Sent Events (`/api/experiments/{id}/events`), polling fallback |
| Versioning + reproducibility checks | "Edit and run new version" records a diff; "Reproduce" re-runs and compares fingerprints |
| Downloadable research report | `/api/experiments/{id}/report` – self-contained HTML, prints to PDF |
| Leaderboard that rewards robustness, not training accuracy | Robustness score, with a "naive rank" column showing where overfit runs would have ranked |

### Brownie points

| Brownie point | How to show it |
|---|---|
| Signal good on training data, fails on unseen data, and why | **Overfit Trap** preset: 100% train accuracy, ~51% test. The hero chart shows the curve rising in training and falling after the dashed line; diagnostics explain why (unlimited depth, 1-day leaves, 390 training days, 24 features) |
| Prevent future information leaking | Truncation test on every feature (recompute on history cut at day k; a causal feature must not change). Leaky features are blocked; the **Leakage Demo** preset shows them caught and disqualified. Plus purging, scaler fitted on train only, and positions decided at the previous close |
| Realistic transaction costs | Commission + slippage on every trade; gross vs net shown side by side; runs under 5 bps are disqualified (**Same RF, zero costs** preset) |
| Predictive model vs simple benchmark | Majority-class baseline and buy-and-hold on every run |
| Reproduce saved experiments | **Reproduce** button: same config hash, dataset SHA-256, prediction and metric fingerprints, all checked |
| Compare across stocks or time periods | Walk-forward folds, regime table, unseen-asset table, Compare page |
| Handle malformed data gracefully | Upload `sample_data/bad/*.csv`: clear errors for unusable files, listed fixes for messy ones. Real Kaggle data: splits/bonuses adjusted from Prev Close, renamed tickers merged |
| Distinguish accuracy, returns and risk-adjusted performance | Three separate panels: "Does it predict?", "Does trading it make money?", "Is it worth the risk?" |
| Report with assumptions, methodology and limitations | Download report |

---

## Five-minute demo script

0. Open the **NSE stocks (real data)** project (it opens first). The synthetic demo project is still there.
1. **Leaderboard** (home): "We rank by robustness, not training accuracy. Look at the naive-rank column:
   by training accuracy, the Overfit Trap would be #1. We put it last."
2. **Open Overfit Trap**: point at the dashed line. "Left of the line the model has seen the answers.
   100% train accuracy, 51% on unseen data. The panel on the left explains why."
3. **Open Leakage Demo**: "72% accuracy, Sharpe 5. Too good to be true, and our truncation test proves
   the centered moving average reads 5 future days. Disqualified automatically."
4. **Open Baseline LR**: the three questions panel, Sharpe confidence interval, costs eating most of the
   gross return, walk-forward folds, regimes, unseen assets.
5. **Click Reproduce**: live progress bar (SSE), then "Reproduced exactly" with all fingerprints matching.
6. **Datasets → upload `BROKEN_no_close_column.csv`**, then `MESSY_but_usable.csv`: errors vs listed fixes.
7. **Edit and run new version** (change C), then **Compare**: diff of settings, overlaid equity curves.
8. **Download report**.

---

## Project layout

```
backend/
  main.py       API routes, background jobs, SSE, demo seeding
  engine.py     split, purge, models, backtest, walk-forward, regimes, diagnostics, robustness score
  features.py   feature registry + look-ahead truncation test
  data.py       CSV validation and cleaning
  report.py     downloadable HTML report
  db.py         SQLite
frontend/       index.html, app.js, styles.css, vendor/chart.umd.js (bundled, works offline)
sample_data/    synthetic DEMO_* assets + broken files for validation demos
scripts/        generate_sample_data.py, fetch_yfinance.py
```

The demo data is synthetic (regime-switching returns with a weak momentum effect) and labelled `DEMO_*`.
It is not real market data. This is research software, not investment advice.
