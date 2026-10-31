"""Head-to-head model comparison on one stock, using SignalForge's own engine for every step.

Same data, features, split, purging, trading rule and costs for every model; only the model changes.
Run:  pip install xgboost  &&  python scripts/model_comparison.py [SYMBOL]
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import engine  # noqa: E402
from backend.data import parse_upload  # noqa: E402
from backend.features import DEFAULT_FEATURES  # noqa: E402

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
SEED = 42


def models():
    out = {
        "Logistic regression (C=0.1)": LogisticRegression(C=0.1, max_iter=2000, random_state=SEED),
        "Random forest, regularized (depth 4, leaf 40)": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=40, random_state=SEED, n_jobs=1),
        "Random forest, unrestricted (sklearn defaults)": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
        "Gradient boosting, sklearn (depth 2, lr 0.05)": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        out["XGBoost (depth 2, lr 0.05, 150 trees)"] = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                                     colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbosity=0)
        out["XGBoost, library defaults (depth 6)"] = XGBClassifier(random_state=SEED, n_jobs=1, verbosity=0)
    except ImportError:
        print("xgboost not installed; skipping XGBoost rows")
    return out


def main():
    archive = ROOT / "sample_data/real/NIFTY50_kaggle_archive.zip"
    df = next(i["df"] for i in parse_upload(archive.read_bytes(), archive.name) if i.get("name") == SYMBOL)
    cfg = engine.normalize_config({"dataset_id": 1, "features": DEFAULT_FEATURES, "model": {"type": "logistic_regression", "params": {}}})
    frame = engine.prepare(df, cfg)
    train, test, info = engine.split_frame(frame, cfg)
    feats = cfg["features"]
    bh = engine.backtest(test["close"], None, cfg, buy_and_hold=True)["stats"]
    base = max(test["y"].mean(), 1 - test["y"].mean())
    n = len(test)
    print(f"{SYMBOL}: train {info['train_start'].date()} → {info['train_end'].date()} ({len(train)} days), "
          f"test {info['test_start'].date()} → {info['test_end'].date()} ({n} days)")
    print(f"Always-guess-majority accuracy {base:.1%}; buy-and-hold Sharpe {bh['sharpe']:.2f}; "
          f"95% margin on any accuracy here ≈ ±{1.96 * math.sqrt(0.25 / n):.1%}\n")

    rows = []
    original = engine.make_model
    for name, clf in models().items():
        engine.make_model = lambda *a, clf=clf: Pipeline([("scaler", StandardScaler()), ("model", clf.__class__(**clf.get_params()))])
        m = engine.make_model()
        m.fit(train[feats], train["y"].astype(int))
        p_tr, p_te = m.predict_proba(train[feats])[:, 1], m.predict_proba(test[feats])[:, 1]
        tr = engine.predictive_metrics(train["y"].values, p_tr)
        te = engine.predictive_metrics(test["y"].values, p_te)
        st = engine.backtest(test["close"], pd.Series(p_te, index=test.index), cfg)["stats"]
        wf = engine.walk_forward(frame, cfg, lambda *_: None).get("summary", {})
        rows.append({"model": name, "train_acc": tr["accuracy"], "test_acc": te["accuracy"], "test_auc": te["roc_auc"],
                     "gap": tr["accuracy"] - te["accuracy"], "test_sharpe": st["sharpe"], "trades": st["n_trades"],
                     "wf_mean_acc": wf.get("mean_accuracy"), "wf_mean_sharpe": wf.get("mean_sharpe"), "wf_std_sharpe": wf.get("std_sharpe")})
    engine.make_model = original
    out = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
    out.to_csv(ROOT / "scripts" / f"model_comparison_{SYMBOL}.csv")


if __name__ == "__main__":
    main()
