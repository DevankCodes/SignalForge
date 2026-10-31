"""SignalForge experiment engine.

Pipeline for one experiment (all deterministic given config + dataset + seed):
  1. load data            2. features + target         3. look-ahead truncation test
  4. chronological split with purging (no target overlap across the boundary)
  5. fit model on TRAIN ONLY (scaler is inside the pipeline, so it never sees test data)
  6. predictive metrics vs majority-class baseline
  7. backtest: signal decided at close t -> position held on day t+1 -> costs on every change
  8. buy-and-hold benchmark on the same window
  9. walk-forward evaluation  10. regime breakdown  11. generalization to other assets
  12. diagnostics + robustness score
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import (ALL_FEATURES, LEAKY_FEATURES, VOLUME_FEATURES, compute_features,
                       compute_target, detect_lookahead)

TRADING_DAYS = 252

MODELS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "params": {"C": {"type": "float", "default": 1.0, "min": 0.0001, "max": 1000, "help": "Inverse regularization strength"}},
    },
    "random_forest": {
        "label": "Random Forest",
        "params": {
            "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 1000, "help": "Number of trees"},
            "max_depth": {"type": "int", "default": 5, "min": 1, "max": 50, "nullable": True, "help": "Empty = unlimited (overfits)"},
            "min_samples_leaf": {"type": "int", "default": 20, "min": 1, "max": 500, "help": "Larger = smoother model"},
        },
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "params": {
            "n_estimators": {"type": "int", "default": 150, "min": 10, "max": 1000, "help": "Boosting rounds"},
            "max_depth": {"type": "int", "default": 2, "min": 1, "max": 10, "help": "Tree depth"},
            "learning_rate": {"type": "float", "default": 0.05, "min": 0.001, "max": 1.0, "help": "Step size"},
            "subsample": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "help": "Row sampling per round"},
        },
    },
    "decision_tree": {
        "label": "Decision Tree",
        "params": {
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 50, "nullable": True, "help": "Empty = unlimited (overfits)"},
            "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 500, "help": "1 = memorise every day"},
        },
    },
}

DEFAULT_CONFIG = {
    "name": "Untitled experiment",
    "dataset_id": None,
    "features": [],
    "model": {"type": "logistic_regression", "params": {}},
    "target": {"horizon": 1, "threshold": 0.0},
    "split": {"mode": "fraction", "train_fraction": 0.7, "train_start": None, "train_end": None, "test_start": None, "test_end": None},
    "embargo_days": 0,
    "signal": {"mode": "long_only", "prob_threshold": 0.5},
    "costs": {"commission_bps": 10.0, "slippage_bps": 5.0},
    "walk_forward": {"enabled": True, "n_folds": 5, "window": "expanding", "window_days": 756},
    "generalization_dataset_ids": [],
    "allow_lookahead": False,
    "seed": 42,
}


class ConfigError(Exception):
    """Invalid experiment configuration; message is shown to the user."""


# --------------------------------------------------------------------------- helpers

def _merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def clean_json(x):
    """Make results JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return x


def canonical_config(config: dict) -> dict:
    keep = {k: config[k] for k in DEFAULT_CONFIG if k in config and k != "name"}
    return json.loads(json.dumps(keep, sort_keys=True))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(canonical_config(config), sort_keys=True).encode()).hexdigest()


def normalize_config(user_config: dict, available_dataset_ids: set[int] | None = None, dataset_has_volume: bool = True) -> dict:
    """Fill defaults and validate. Raises ConfigError with a human-readable message."""
    cfg = _merge(DEFAULT_CONFIG, user_config or {})
    problems = []

    if cfg["dataset_id"] is None:
        problems.append("Choose a dataset to train on.")
    elif available_dataset_ids is not None and int(cfg["dataset_id"]) not in available_dataset_ids:
        problems.append(f"Dataset #{cfg['dataset_id']} does not exist in this project.")

    feats = cfg["features"]
    if not feats:
        problems.append("Select at least one feature.")
    unknown = [f for f in feats if f not in ALL_FEATURES]
    if unknown:
        problems.append(f"Unknown feature(s): {', '.join(unknown)}.")
    if len(set(feats)) != len(feats):
        cfg["features"] = list(dict.fromkeys(feats))
    if not dataset_has_volume:
        vol = [f for f in feats if f in VOLUME_FEATURES]
        if vol:
            problems.append(f"This dataset has no volume column, so these features cannot be computed: {', '.join(vol)}.")

    mtype = cfg["model"].get("type")
    if mtype not in MODELS:
        problems.append(f"Unknown model type '{mtype}'. Choose one of: {', '.join(MODELS)}.")
    else:
        spec = MODELS[mtype]["params"]
        params = {}
        for pname, pspec in spec.items():
            val = cfg["model"].get("params", {}).get(pname, pspec["default"])
            if val in ("", None):
                if pspec.get("nullable") or pspec["default"] is None:
                    params[pname] = None
                    continue
                val = pspec["default"]
            try:
                val = int(val) if pspec["type"] == "int" else float(val)
            except (TypeError, ValueError):
                problems.append(f"Model parameter '{pname}' must be a number.")
                continue
            if not (pspec["min"] <= val <= pspec["max"]):
                problems.append(f"Model parameter '{pname}' must be between {pspec['min']} and {pspec['max']} (got {val}).")
            params[pname] = val
        cfg["model"]["params"] = params

    h = cfg["target"].get("horizon")
    if not isinstance(h, int) or not (1 <= h <= 20):
        try:
            h = int(h)
            cfg["target"]["horizon"] = h
            if not 1 <= h <= 20:
                raise ValueError
        except (TypeError, ValueError):
            problems.append("Target horizon must be a whole number of days between 1 and 20.")
    try:
        cfg["target"]["threshold"] = float(cfg["target"].get("threshold", 0.0))
        if abs(cfg["target"]["threshold"]) > 0.2:
            problems.append("Target threshold must be between -0.2 and 0.2 (e.g. 0.0 = 'price goes up').")
    except (TypeError, ValueError):
        problems.append("Target threshold must be a number.")

    sp = cfg["split"]
    if sp.get("mode") not in ("fraction", "date"):
        problems.append("Split mode must be 'fraction' or 'date'.")
    if sp.get("mode") == "fraction":
        try:
            tf = float(sp.get("train_fraction", 0.7))
            sp["train_fraction"] = tf
            if not 0.3 <= tf <= 0.9:
                problems.append("Train fraction must be between 0.3 and 0.9 so both periods have enough data.")
        except (TypeError, ValueError):
            problems.append("Train fraction must be a number.")
    else:
        for key in ("train_end",):
            if not sp.get(key):
                problems.append("Date split needs at least a train end date.")
        try:
            dts = {k: pd.Timestamp(sp[k]) for k in ("train_start", "train_end", "test_start", "test_end") if sp.get(k)}
            if "train_start" in dts and "train_end" in dts and dts["train_start"] >= dts["train_end"]:
                problems.append("Train start must be before train end.")
            if "test_start" in dts and "train_end" in dts and dts["test_start"] <= dts["train_end"]:
                problems.append("Test period must start after the training period ends (chronological split).")
            if "test_start" in dts and "test_end" in dts and dts["test_start"] >= dts["test_end"]:
                problems.append("Test start must be before test end.")
        except Exception:
            problems.append("Split dates must be valid dates (YYYY-MM-DD).")

    try:
        cfg["embargo_days"] = int(cfg.get("embargo_days") or 0)
        if not 0 <= cfg["embargo_days"] <= 60:
            problems.append("Embargo must be between 0 and 60 days.")
    except (TypeError, ValueError):
        problems.append("Embargo must be a whole number of days.")

    if cfg["signal"].get("mode") not in ("long_only", "long_short"):
        problems.append("Signal mode must be 'long_only' or 'long_short'.")
    try:
        pt = float(cfg["signal"].get("prob_threshold", 0.5))
        cfg["signal"]["prob_threshold"] = pt
        if not 0.3 <= pt <= 0.8:
            problems.append("Probability threshold must be between 0.3 and 0.8.")
    except (TypeError, ValueError):
        problems.append("Probability threshold must be a number.")

    for key in ("commission_bps", "slippage_bps"):
        try:
            v = float(cfg["costs"].get(key, 0))
            cfg["costs"][key] = v
            if not 0 <= v <= 200:
                problems.append(f"{key.replace('_', ' ').title()} must be between 0 and 200 basis points.")
        except (TypeError, ValueError):
            problems.append(f"{key} must be a number.")

    wf = cfg["walk_forward"]
    try:
        wf["n_folds"] = int(wf.get("n_folds", 5))
        if wf.get("enabled") and not 2 <= wf["n_folds"] <= 12:
            problems.append("Walk-forward folds must be between 2 and 12.")
        wf["window_days"] = int(wf.get("window_days", 756))
        if wf.get("window") not in ("expanding", "rolling"):
            problems.append("Walk-forward window must be 'expanding' or 'rolling'.")
        if wf.get("window") == "rolling" and wf["window_days"] < 120:
            problems.append("Rolling window must be at least 120 trading days.")
    except (TypeError, ValueError):
        problems.append("Walk-forward settings must be whole numbers.")

    leaky_selected = [f for f in cfg["features"] if f in LEAKY_FEATURES]
    if leaky_selected and not cfg["allow_lookahead"]:
        problems.append(
            "Feature(s) " + ", ".join(leaky_selected) + " use future data. They are blocked by the leakage guard. "
            "Remove them, or tick 'Allow look-ahead (demo only)' to see what leakage does to results."
        )

    try:
        cfg["seed"] = int(cfg.get("seed", 42))
    except (TypeError, ValueError):
        problems.append("Seed must be a whole number.")

    gen = [int(x) for x in cfg.get("generalization_dataset_ids") or []]
    cfg["generalization_dataset_ids"] = [g for g in dict.fromkeys(gen) if g != cfg.get("dataset_id")]

    if problems:
        raise ConfigError(" ".join(problems))
    return cfg


def make_model(mtype: str, params: dict, seed: int) -> Pipeline:
    if mtype == "logistic_regression":
        clf = LogisticRegression(C=params.get("C", 1.0), max_iter=2000, random_state=seed)
    elif mtype == "random_forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params.get("max_depth"),
                                     min_samples_leaf=params["min_samples_leaf"], random_state=seed, n_jobs=1)
    elif mtype == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                         learning_rate=params["learning_rate"], subsample=params["subsample"],
                                         random_state=seed)
    elif mtype == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=params.get("max_depth"), min_samples_leaf=params["min_samples_leaf"],
                                     random_state=seed)
    else:
        raise ConfigError(f"Unknown model '{mtype}'")
    # Scaler lives inside the pipeline: fitted on training rows only, so test statistics never leak in.
    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def _proba_up(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    p = model.predict_proba(X)
    if 1.0 in classes:
        return p[:, classes.index(1.0)]
    return np.zeros(len(X))


# --------------------------------------------------------------------------- metrics

def predictive_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if len(set(y_true)) > 1 else float("nan"),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "base_rate_up": base_rate,
        "majority_baseline_accuracy": max(base_rate, 1 - base_rate),
        "pct_predicted_up": float(pred.mean()),
    }
    out["edge_vs_majority"] = out["accuracy"] - out["majority_baseline_accuracy"]
    return out


def backtest(close: pd.Series, proba: pd.Series | None, cfg: dict, buy_and_hold: bool = False) -> dict:
    """Daily backtest. `close` and `proba` share a DatetimeIndex covering one evaluation window.

    Timing (no look-ahead): the prediction made with data up to the close of day t sets the position
    held over day t+1, i.e. strategy_return[t+1] = position[t] * asset_return[t+1].
    Costs = (commission + slippage) * |change in position| on every rebalance, entry and exit included.
    """
    ret = close.pct_change().fillna(0.0)
    if buy_and_hold:
        signal = pd.Series(1.0, index=close.index)
    else:
        thr = cfg["signal"]["prob_threshold"]
        low = -1.0 if cfg["signal"]["mode"] == "long_short" else 0.0
        signal = pd.Series(np.where(proba.values >= thr, 1.0, low), index=close.index)
    position = signal.shift(1).fillna(0.0)          # decided yesterday, held today
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost_rate = (cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]) / 1e4
    costs = turnover * cost_rate
    gross = position * ret
    net = gross - costs
    stats = perf_stats(net, position, costs, gross)
    return {"net": net, "gross": gross, "position": position, "stats": stats}


def perf_stats(net: pd.Series, position: pd.Series, costs: pd.Series, gross: pd.Series) -> dict:
    r = net.iloc[1:] if len(net) > 1 else net  # first day has no return
    n = len(r)
    if n == 0:
        return {}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    sd = r.std(ddof=0)
    sharpe = float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    downside = math.sqrt(float((np.minimum(r, 0) ** 2).mean()))
    sortino = float(r.mean() * TRADING_DAYS / (downside * math.sqrt(TRADING_DAYS))) if downside > 1e-12 else 0.0
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    in_mkt = position.iloc[1:] != 0
    g = gross.iloc[1:]
    gross_eq = float((1 + g).prod() - 1)
    gsd = g.std(ddof=0)
    # Standard error of the annualised Sharpe ratio (Lo, 2002, iid approximation).
    sharpe_se = math.sqrt((1 + 0.5 * sharpe ** 2) / years) if years > 0 else float("nan")
    return {
        "days": n,
        "sharpe_ci95": [sharpe - 1.96 * sharpe_se, sharpe + 1.96 * sharpe_se],
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": float(sd * math.sqrt(TRADING_DAYS)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else None,
        "exposure": float(in_mkt.mean()),
        "n_trades": int((position.diff().fillna(position) != 0).sum()),
        "win_rate_in_market": float((r[in_mkt] > 0).mean()) if in_mkt.any() else None,
        "total_costs": float(costs.sum()),
        "gross_total_return": gross_eq,
        "gross_sharpe": float(g.mean() / gsd * math.sqrt(TRADING_DAYS)) if gsd > 1e-12 else 0.0,
    }


def _curve(dates: pd.Index, net: pd.Series, max_points: int = 900) -> dict:
    eq = (1 + net).cumprod()
    dd = eq / eq.cummax() - 1
    step = max(1, int(math.ceil(len(eq) / max_points)))
    idx = list(range(0, len(eq), step))
    if idx and idx[-1] != len(eq) - 1:
        idx.append(len(eq) - 1)
    return {
        "dates": [dates[i].strftime("%Y-%m-%d") for i in idx],
        "equity": [round(float(eq.iloc[i]), 5) for i in idx],
        "drawdown": [round(float(dd.iloc[i]), 5) for i in idx],
    }


# --------------------------------------------------------------------------- data prep

def prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Features + target on the full history, then drop rows with any NaN (warm-up / last h rows)."""
    feats = compute_features(df, cfg["features"])
    y, fwd = compute_target(df, cfg["target"]["horizon"], cfg["target"]["threshold"])
    frame = feats.copy()
    frame["y"] = y
    frame["fwd_ret"] = fwd
    frame["close"] = df["close"].values
    frame["date"] = df["date"].values
    frame = frame.dropna(subset=cfg["features"] + ["y"]).set_index("date")
    return frame


def split_frame(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sp, h = cfg["split"], cfg["target"]["horizon"]
    if sp["mode"] == "fraction":
        cut = int(len(frame) * sp["train_fraction"])
        train, test = frame.iloc[:cut], frame.iloc[cut:]
        if sp.get("train_start"):
            train = train[train.index >= pd.Timestamp(sp["train_start"])]
    else:
        train_end = pd.Timestamp(sp["train_end"])
        train = frame[frame.index <= train_end]
        if sp.get("train_start"):
            train = train[train.index >= pd.Timestamp(sp["train_start"])]
        test_start = pd.Timestamp(sp["test_start"]) if sp.get("test_start") else train_end + pd.Timedelta(days=1)
        test = frame[frame.index >= test_start]
        if sp.get("test_end"):
            test = test[test.index <= pd.Timestamp(sp["test_end"])]
    # Purge: the last h training labels look h days ahead, into the test period. Drop them.
    purged = min(h, len(train))
    train = train.iloc[: len(train) - purged] if purged else train
    embargo = cfg.get("embargo_days", 0)
    if embargo:
        test = test.iloc[embargo:]
    info = {"purged_rows": purged, "embargo_rows": embargo}
    if len(train) < 100:
        raise ConfigError(f"Training period has only {len(train)} usable rows after feature warm-up; need at least 100. "
                          "Use an earlier train start or a larger train fraction.")
    if len(test) < 40:
        raise ConfigError(f"Test period has only {len(test)} usable rows; need at least 40. Move the split earlier.")
    if train["y"].nunique() < 2:
        raise ConfigError("Training labels contain only one class; the model cannot learn. Try a different threshold.")
    info.update({
        "train_start": train.index[0], "train_end": train.index[-1], "train_rows": len(train),
        "test_start": test.index[0], "test_end": test.index[-1], "test_rows": len(test),
    })
    return train, test, info


# --------------------------------------------------------------------------- analyses

def feature_importance(model: Pipeline, names: list[str]) -> list[dict]:
    clf = model.named_steps["model"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        kind = "impurity importance"
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
        kind = "|standardized coefficient|"
    else:
        return []
    total = float(np.sum(vals)) or 1.0
    rows = [{"feature": n, "importance": float(v) / total, "kind": kind} for n, v in zip(names, vals)]
    if hasattr(clf, "coef_"):
        for row, c in zip(rows, clf.coef_[0]):
            row["direction"] = "bullish" if c > 0 else "bearish"
    return sorted(rows, key=lambda r: -r["importance"])


def walk_forward(frame: pd.DataFrame, cfg: dict, progress: Callable[[float, str], None]) -> dict:
    wf, h = cfg["walk_forward"], cfg["target"]["horizon"]
    n, k = len(frame), wf["n_folds"]
    min_train = max(252, int(n * 0.3))
    if n - min_train < k * 40:
        return {"enabled": True, "skipped": f"Not enough data for {k} folds of at least 40 days after a {min_train}-day initial training window."}
    fold_size = (n - min_train) // k
    feats = cfg["features"]
    folds, oos_parts, bh_parts = [], [], []
    for i in range(k):
        t0 = min_train + i * fold_size
        t1 = n if i == k - 1 else t0 + fold_size
        tr_end = t0 - h  # purge
        tr_begin = 0 if wf["window"] == "expanding" else max(0, tr_end - wf["window_days"])
        tr, te = frame.iloc[tr_begin:tr_end], frame.iloc[t0:t1]
        if tr["y"].nunique() < 2:
            continue
        model = make_model(cfg["model"]["type"], cfg["model"]["params"], cfg["seed"])
        model.fit(tr[feats], tr["y"])
        p = _proba_up(model, te[feats])
        pm = predictive_metrics(te["y"].values, p, cfg["signal"]["prob_threshold"])
        bt = backtest(te["close"], pd.Series(p, index=te.index), cfg)
        bh = backtest(te["close"], None, cfg, buy_and_hold=True)
        train_acc = accuracy_score(tr["y"], (_proba_up(model, tr[feats]) >= cfg["signal"]["prob_threshold"]).astype(int))
        folds.append({
            "fold": i + 1,
            "train_start": tr.index[0], "train_end": tr.index[-1], "train_rows": len(tr),
            "test_start": te.index[0], "test_end": te.index[-1], "test_rows": len(te),
            "train_accuracy": train_acc, "accuracy": pm["accuracy"], "roc_auc": pm["roc_auc"],
            "edge_vs_majority": pm["edge_vs_majority"],
            "strategy_return": bt["stats"]["total_return"], "strategy_sharpe": bt["stats"]["sharpe"],
            "max_drawdown": bt["stats"]["max_drawdown"],
            "buy_hold_return": bh["stats"]["total_return"], "buy_hold_sharpe": bh["stats"]["sharpe"],
            "beats_buy_hold": bt["stats"]["sharpe"] > bh["stats"]["sharpe"],
        })
        oos_parts.append(bt["net"])
        bh_parts.append(bh["net"])
        progress(0.62 + 0.18 * (i + 1) / k, f"Walk-forward fold {i + 1}/{k}")
    if not folds:
        return {"enabled": True, "skipped": "No fold had both classes in its training data."}
    sharpes = np.array([f["strategy_sharpe"] for f in folds])
    oos = pd.concat(oos_parts)
    bh_all = pd.concat(bh_parts)
    return {
        "enabled": True,
        "window": wf["window"],
        "folds": folds,
        "summary": {
            "mean_accuracy": float(np.mean([f["accuracy"] for f in folds])),
            "mean_auc": float(np.nanmean([f["roc_auc"] if f["roc_auc"] is not None else np.nan for f in folds])),
            "mean_sharpe": float(sharpes.mean()),
            "std_sharpe": float(sharpes.std(ddof=0)),
            "min_sharpe": float(sharpes.min()),
            "pct_folds_positive": float((sharpes > 0).mean()),
            "pct_folds_beat_buy_hold": float(np.mean([f["beats_buy_hold"] for f in folds])),
        },
        "curve": _curve(oos.index, oos),
        "buy_hold_curve": _curve(bh_all.index, bh_all),
    }


def regime_breakdown(df: pd.DataFrame, test: pd.DataFrame, net: pd.Series, bh_net: pd.Series,
                     proba: np.ndarray, train_index: pd.Index, cfg: dict) -> list[dict]:
    """Label each test day by market regime using ONLY past data, then compare performance per regime."""
    close = df.set_index("date")["close"]
    sma200 = close.rolling(200).mean()
    vol20 = close.pct_change().rolling(20).std()
    vol_cut = float(vol20.reindex(train_index).median())  # threshold learnt on training period only
    # Regime known at yesterday's close: that is when the position was chosen.
    trend = np.where(close > sma200, "Bull (above 200-day MA)", "Bear (below 200-day MA)")
    trend = pd.Series(trend, index=close.index).where(sma200.notna()).shift(1)
    volr = pd.Series(np.where(vol20 > vol_cut, "High volatility", "Low volatility"), index=close.index).where(vol20.notna()).shift(1)
    pred = pd.Series((proba >= cfg["signal"]["prob_threshold"]).astype(int), index=test.index)
    rows = []
    for family, labels in (("Trend", trend), ("Volatility", volr)):
        lab = labels.reindex(test.index)
        for name in sorted(lab.dropna().unique()):
            mask = lab == name
            if mask.sum() < 15:
                continue
            r, b = net[mask], bh_net[mask]
            sd, bsd = r.std(ddof=0), b.std(ddof=0)
            rows.append({
                "family": family, "regime": name, "days": int(mask.sum()),
                "accuracy": float((pred[mask] == test["y"][mask]).mean()),
                "strategy_ann_return": float(r.mean() * TRADING_DAYS),
                "strategy_sharpe": float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0,
                "buy_hold_ann_return": float(b.mean() * TRADING_DAYS),
                "buy_hold_sharpe": float(b.mean() / bsd * math.sqrt(TRADING_DAYS)) if bsd > 1e-12 else 0.0,
            })
    return rows


def diagnose(res: dict, cfg: dict) -> list[dict]:
    """Deterministic, plain-English findings. Each has a level: good / warn / bad / info."""
    out = []
    tr, te = res["predictive"]["train"], res["predictive"]["test"]
    gap = tr["accuracy"] - te["accuracy"]
    s_tr, s_te = res["strategy"]["train"], res["strategy"]["test"]
    bh = res["benchmark"]["test"]
    leaks = [f for f, v in res["leakage"]["features"].items() if v["leaky"]]

    if leaks:
        out.append({"level": "bad", "title": "Look-ahead leakage detected",
                    "detail": f"The truncation test shows {', '.join(leaks)} change value when future rows are added, so they use information "
                              "that would not exist at prediction time. Every metric in this run is invalid; it is excluded from the leaderboard."})
    if gap > 0.10:
        reasons = []
        p = cfg["model"]["params"]
        if p.get("max_depth") is None and cfg["model"]["type"] in ("random_forest", "decision_tree"):
            reasons.append("trees have unlimited depth")
        if p.get("min_samples_leaf", 99) <= 2:
            reasons.append(f"min_samples_leaf={p.get('min_samples_leaf')} lets a leaf hold a single day")
        if res["split"]["train_rows"] < 500:
            reasons.append(f"only {res['split']['train_rows']} training days")
        if len(cfg["features"]) > 12:
            reasons.append(f"{len(cfg['features'])} features give the model many ways to fit noise")
        why = "; ".join(reasons) if reasons else "the model is flexible relative to the weak signal in daily returns"
        out.append({"level": "bad", "title": "Overfitting: great on training data, fails on unseen data",
                    "detail": f"Train accuracy {tr['accuracy']:.1%} vs test {te['accuracy']:.1%} (gap {gap:.1%}). In-sample Sharpe {s_tr['sharpe']:.2f} "
                              f"vs out-of-sample {s_te['sharpe']:.2f}. Why: {why}. Daily returns are mostly noise, so a model that memorises "
                              "the training days learns patterns that never repeat."})
    elif gap > 0.05:
        out.append({"level": "warn", "title": "Mild overfitting",
                    "detail": f"Train accuracy {tr['accuracy']:.1%} vs test {te['accuracy']:.1%}. Consider stronger regularization or fewer features."})
    else:
        out.append({"level": "good", "title": "Train and test behave alike",
                    "detail": f"Train accuracy {tr['accuracy']:.1%} vs test {te['accuracy']:.1%}: no sign of memorisation."})

    edge = te["edge_vs_majority"]
    auc = te["roc_auc"]
    if auc is not None and not math.isnan(auc) and abs(auc - 0.5) < 0.02 and edge <= 0.01:
        out.append({"level": "warn", "title": "No predictive edge on the test set",
                    "detail": f"Test AUC {auc:.3f} is close to 0.5 and accuracy beats the always-predict-the-majority baseline by {edge:+.1%}. "
                              "The signal is indistinguishable from a coin flip."})
    elif edge > 0.01:
        out.append({"level": "good", "title": "Beats the naive predictor",
                    "detail": f"Test accuracy beats always-predict-the-majority-class by {edge:+.1%} (AUC {auc:.3f})."})

    if te["accuracy"] > 0.65 or s_te["sharpe"] > 3:
        out.append({"level": "warn", "title": "Too good to be true?",
                    "detail": f"Test accuracy {te['accuracy']:.1%} / Sharpe {s_te['sharpe']:.2f} is far above what daily equity signals usually "
                              "achieve. Double-check for leakage before trusting this."})

    cost_bps = cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]
    if cost_bps < MIN_REALISTIC_COST_BPS:
        out.append({"level": "bad", "title": "Unrealistic transaction costs",
                    "detail": f"Costs are set to {cost_bps:g} bps per trade. Real brokerage, taxes, spreads and slippage on NSE equities are "
                              f"typically well above {MIN_REALISTIC_COST_BPS:g} bps, so this backtest overstates returns. Excluded from the leaderboard."})
    cost_drag = s_te["gross_total_return"] - s_te["total_return"]
    if s_te["n_trades"] > 0:
        lvl = "warn" if s_te["gross_sharpe"] > 0 and s_te["sharpe"] <= 0 else "info"
        out.append({"level": lvl, "title": "Transaction-cost impact",
                    "detail": f"{s_te['n_trades']} position changes cost {cost_drag:.1%} of return. Sharpe before costs {s_te['gross_sharpe']:.2f}, "
                              f"after costs {s_te['sharpe']:.2f}." + (" Costs turn a profitable signal into a losing one." if lvl == "warn" else "")})

    lo, hi = s_te["sharpe_ci95"]
    if lo <= 0 <= hi:
        out.append({"level": "info", "title": "Not statistically significant",
                    "detail": f"Test Sharpe {s_te['sharpe']:.2f} has a 95% confidence interval of [{lo:.2f}, {hi:.2f}] over "
                              f"{s_te['days']} days, which includes zero. The result could be luck; more test data is needed to be sure."})
    elif lo > 0:
        out.append({"level": "good", "title": "Statistically significant Sharpe",
                    "detail": f"95% confidence interval for test Sharpe is [{lo:.2f}, {hi:.2f}], entirely above zero."})

    if s_te["sharpe"] > bh["sharpe"]:
        out.append({"level": "good", "title": "Beats buy-and-hold (risk-adjusted)",
                    "detail": f"Strategy Sharpe {s_te['sharpe']:.2f} vs buy-and-hold {bh['sharpe']:.2f}; max drawdown {s_te['max_drawdown']:.1%} vs {bh['max_drawdown']:.1%}."})
    else:
        out.append({"level": "warn", "title": "Does not beat buy-and-hold",
                    "detail": f"Strategy Sharpe {s_te['sharpe']:.2f} vs buy-and-hold {bh['sharpe']:.2f}. Simply holding the asset was better risk-adjusted."})

    wfs = res.get("walk_forward", {}).get("summary")
    if wfs:
        lvl = "good" if wfs["mean_sharpe"] > 0 and wfs["pct_folds_positive"] >= 0.6 and wfs["std_sharpe"] < max(1.0, abs(wfs["mean_sharpe"]) * 1.5) else "warn"
        out.append({"level": lvl, "title": "Stability across time (walk-forward)",
                    "detail": f"Sharpe was positive in {wfs['pct_folds_positive']:.0%} of folds (mean {wfs['mean_sharpe']:.2f}, std {wfs['std_sharpe']:.2f}) "
                              f"and beat buy-and-hold in {wfs['pct_folds_beat_buy_hold']:.0%}."})
    gen = res.get("generalization") or []
    if gen:
        good = [g for g in gen if g.get("strategy_sharpe", 0) > g.get("buy_hold_sharpe", 0)]
        out.append({"level": "good" if len(good) >= len(gen) / 2 else "warn", "title": "Generalization to unseen assets",
                    "detail": f"Without retraining, the model beat buy-and-hold on {len(good)} of {len(gen)} other asset(s)."})
    return out


MIN_REALISTIC_COST_BPS = 5.0  # commission + slippage per unit traded; below this a backtest is not credible


def robustness_score(res: dict, cost_bps: float = 15.0) -> dict:
    """Leaderboard score that rewards out-of-sample, stable performance, not training accuracy.

    score = OOS Sharpe (walk-forward mean if available, else test Sharpe)
            - 0.5 * Sharpe std across folds        (penalise instability; Sharpe standard error if no walk-forward)
            - 3.0 * max(0, train acc - test acc)   (penalise overfitting)
            + 0.5 * (Sharpe - buy & hold Sharpe), capped at +/-1   (reward beating the benchmark)
    Leaky experiments and experiments with unrealistically low costs are disqualified.
    """
    te_sharpe = res["strategy"]["test"]["sharpe"]
    wfs = res.get("walk_forward", {}).get("summary")
    oos = wfs["mean_sharpe"] if wfs else te_sharpe
    # Without walk-forward we cannot measure stability across time, so use the Sharpe standard error instead.
    ci = res["strategy"]["test"].get("sharpe_ci95") or [te_sharpe, te_sharpe]
    instab = wfs["std_sharpe"] if wfs else (ci[1] - ci[0]) / (2 * 1.96)
    gap = max(0.0, res["predictive"]["train"]["accuracy"] - res["predictive"]["test"]["accuracy"])
    excess = max(-1.0, min(1.0, te_sharpe - res["benchmark"]["test"]["sharpe"]))
    score = oos - 0.5 * instab - 3.0 * gap + 0.5 * excess
    reasons = []
    if any(v["leaky"] for v in res["leakage"]["features"].values()):
        reasons.append("Look-ahead leakage detected")
    if cost_bps < MIN_REALISTIC_COST_BPS:
        reasons.append(f"Unrealistic transaction costs ({cost_bps:g} bps < {MIN_REALISTIC_COST_BPS} bps minimum)")
    return {
        "score": None if reasons else round(score, 4),
        "disqualified": bool(reasons),
        "disqualified_reasons": reasons,
        "raw_score": round(score, 4),
        "components": {"oos_sharpe": oos, "instability_penalty": -0.5 * instab,
                       "overfit_penalty": -3.0 * gap, "benchmark_bonus": 0.5 * excess},
    }


# --------------------------------------------------------------------------- portfolio

def build_portfolio(legs: list[dict], max_weight: float, max_gross: float, cost_bps: float) -> dict:
    """Combine saved single-stock signals into one portfolio under allocation constraints.

    legs: [{"id", "name", "daily": {"dates", "position", "asset_return"}}]
    Rule, each day (positions were already decided at the previous close, so there is no look-ahead):
      active = legs whose signal is long (or short) today
      weight = sign(position) * min(max_weight, max_gross / number active); anything not allocated stays in cash
    Costs = cost_bps per unit of weight traded, every day. Benchmark = equal-weight buy-and-hold of the same stocks.
    """
    if len(legs) < 2:
        raise ConfigError("Pick at least two experiments to combine.")
    if not (0.05 <= max_weight <= 1.0):
        raise ConfigError("Maximum weight per stock must be between 5% and 100%.")
    if not (0.1 <= max_gross <= 1.0):
        raise ConfigError("Maximum total exposure must be between 10% and 100% (no leverage).")
    if not (0 <= cost_bps <= 200):
        raise ConfigError("Costs must be between 0 and 200 basis points.")
    names = [l["name"] for l in legs]
    if len(set(names)) < len(names):
        raise ConfigError("Two of the selected experiments trade the same stock. Pick one signal per stock.")
    pos = pd.concat({l["name"]: pd.Series(l["daily"]["position"], index=pd.to_datetime(l["daily"]["dates"])) for l in legs}, axis=1, join="inner")
    ret = pd.concat({l["name"]: pd.Series(l["daily"]["asset_return"], index=pd.to_datetime(l["daily"]["dates"])) for l in legs}, axis=1, join="inner")
    if len(pos) < 60:
        raise ConfigError(f"The selected experiments share only {len(pos)} test days; at least 60 are needed. "
                          "Use experiments with the same split settings on stocks with overlapping history.")
    active = (pos != 0).sum(axis=1)
    per = np.where(active > 0, np.minimum(max_weight, max_gross / active.clip(lower=1)), 0.0)
    w = np.sign(pos).mul(per, axis=0)
    cap_binding = (active > 0) & (max_gross / active.clip(lower=1) > max_weight)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    rate = cost_bps / 1e4
    gross = (w * ret).sum(axis=1)
    costs = turnover * rate
    net = gross - costs
    exposure = w.abs().sum(axis=1)
    stats = perf_stats(net, exposure.where(exposure > 0, 0), costs, gross)

    bh_w = pd.DataFrame(1.0 / len(legs), index=ret.index, columns=ret.columns)
    bh_net = (bh_w * ret).sum(axis=1)
    bh_cost = pd.Series(0.0, index=ret.index)
    bh_cost.iloc[0] = rate
    bh_net = bh_net - bh_cost
    bh_stats = perf_stats(bh_net, pd.Series(1.0, index=ret.index), bh_cost, bh_net + bh_cost)

    leg_rows, leg_nets = [], {}
    for l in legs:
        p, r = pos[l["name"]], ret[l["name"]]
        t = p.diff().abs()
        t.iloc[0] = abs(p.iloc[0])
        n = p * r - t * rate
        leg_nets[l["name"]] = n
        s = perf_stats(n, p, t * rate, p * r)
        leg_rows.append({"experiment_id": l["id"], "name": l["name"], "sharpe": s["sharpe"], "total_return": s["total_return"],
                         "max_drawdown": s["max_drawdown"], "exposure": s["exposure"],
                         "avg_weight": float(w[l["name"]].abs().mean())})
    corr = pd.DataFrame(leg_nets).corr().values
    off = corr[~np.eye(len(legs), dtype=bool)]
    best = max(leg_rows, key=lambda r: r["sharpe"])
    findings = []
    if stats["sharpe"] > bh_stats["sharpe"]:
        findings.append(f"The portfolio’s Sharpe ({stats['sharpe']:.2f}) beats equal-weight buy-and-hold of the same stocks ({bh_stats['sharpe']:.2f}).")
    else:
        findings.append(f"Equal-weight buy-and-hold of the same stocks has the better Sharpe ({bh_stats['sharpe']:.2f} vs {stats['sharpe']:.2f}).")
    if stats["sharpe"] > best["sharpe"]:
        findings.append(f"Combining helps: the portfolio’s Sharpe is above the best single signal ({best['name']}, {best['sharpe']:.2f}).")
    else:
        findings.append(f"The best single signal ({best['name']}, Sharpe {best['sharpe']:.2f}) beats the combination.")
    findings.append(f"Average correlation between the signals’ daily returns is {np.nanmean(off):.2f}; lower correlation means more diversification.")
    findings.append(f"The {max_weight:.0%} cap per stock was binding on {cap_binding.mean():.0%} of days; average invested share was {exposure.mean():.0%}.")
    return clean_json({
        "start": pos.index[0], "end": pos.index[-1], "days": len(pos),
        "constraints": {"max_weight": max_weight, "max_gross": max_gross, "cost_bps": cost_bps},
        "portfolio": stats, "benchmark": bh_stats, "legs": leg_rows,
        "avg_pairwise_correlation": float(np.nanmean(off)),
        "cap_binding_share": float(cap_binding.mean()), "avg_exposure": float(exposure.mean()),
        "curve": _curve(net.index, net), "benchmark_curve": _curve(bh_net.index, bh_net),
        "findings": findings,
        "rule": "Each day, every stock whose signal is active gets an equal share of the allowed exposure, "
                "capped per stock; the rest stays in cash. Positions come from each experiment’s saved test-period "
                "signals, decided at the previous close.",
    })


# --------------------------------------------------------------------------- main entry

def run_experiment(cfg: dict, load_dataset: Callable[[int], tuple[pd.DataFrame, dict]],
                   progress: Callable[[float, str], None] = lambda p, s: None) -> dict:
    t_start = time.time()
    np.random.seed(cfg["seed"])
    feats = cfg["features"]

    progress(0.03, "Loading dataset")
    df, ds_meta = load_dataset(cfg["dataset_id"])

    progress(0.10, "Checking every feature for look-ahead bias")
    leak = detect_lookahead(df, feats, seed=cfg["seed"])
    leaky = [f for f, v in leak.items() if v["leaky"]]
    if leaky and not cfg["allow_lookahead"]:
        raise ConfigError(f"Leakage guard: {', '.join(leaky)} use future data (truncation test failed). "
                          "Remove them or enable 'Allow look-ahead (demo only)'.")

    progress(0.18, "Engineering features and target")
    frame = prepare(df, cfg)
    train, test, split_info = split_frame(frame, cfg)

    progress(0.28, f"Training {MODELS[cfg['model']['type']]['label']} on {len(train)} days")
    model = make_model(cfg["model"]["type"], cfg["model"]["params"], cfg["seed"])
    model.fit(train[feats], train["y"])

    progress(0.40, "Scoring train and test periods")
    p_train = _proba_up(model, train[feats])
    p_test = _proba_up(model, test[feats])
    thr = cfg["signal"]["prob_threshold"]
    pm_train = predictive_metrics(train["y"].values, p_train, thr)
    pm_test = predictive_metrics(test["y"].values, p_test, thr)

    progress(0.50, "Backtesting with transaction costs")
    bt_train = backtest(train["close"], pd.Series(p_train, index=train.index), cfg)
    bt_test = backtest(test["close"], pd.Series(p_test, index=test.index), cfg)
    bh_train = backtest(train["close"], None, cfg, buy_and_hold=True)
    bh_test = backtest(test["close"], None, cfg, buy_and_hold=True)

    res = {
        "dataset": {"id": cfg["dataset_id"], "name": ds_meta.get("name"), "sha256": ds_meta.get("sha256"),
                    "rows": int(len(df)), "usable_rows": int(len(frame))},
        "split": split_info,
        "leakage": {"features": leak, "method": "Truncation test: each feature is recomputed on history cut at several dates; "
                    "a causal feature must give the same value on the cut date as on the full history."},
        "predictive": {"train": pm_train, "test": pm_test},
        "strategy": {"train": bt_train["stats"], "test": bt_test["stats"]},
        "benchmark": {"name": "Buy and hold", "train": bh_train["stats"], "test": bh_test["stats"]},
        "curves": {
            "test": _curve(test.index, bt_test["net"]), "test_buy_hold": _curve(test.index, bh_test["net"]),
            "train": _curve(train.index, bt_train["net"]), "train_buy_hold": _curve(train.index, bh_train["net"]),
        },
        "feature_importance": feature_importance(model, feats),
    }

    # Daily test-period positions and asset returns: lets saved runs be combined into a portfolio later.
    res["daily_test"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in test.index],
        "position": [float(x) for x in bt_test["position"].values],
        "asset_return": [round(float(x), 10) for x in test["close"].pct_change().fillna(0.0).values],
    }

    # Predictions vs what actually happened (last test days), and the latest signal after the data ends.
    recent = test.iloc[-15:]
    p_recent = p_test[-15:]
    res["recent_predictions"] = [{
        "date": d, "prob_up": float(p), "predicted": "UP" if p >= thr else "DOWN",
        "actual_return": float(r), "actual": "UP" if y == 1 else "DOWN", "correct": bool((p >= thr) == (y == 1)),
    } for d, p, r, y in zip(recent.index, p_recent, recent["fwd_ret"], recent["y"])]
    allf = compute_features(df, feats).assign(date=df["date"].values).dropna(subset=feats)
    if len(allf):
        last = allf.iloc[[-1]]
        p_last = float(_proba_up(model, last[feats])[0])
        res["latest_signal"] = {
            "as_of": pd.Timestamp(last["date"].iloc[0]).strftime("%Y-%m-%d"), "horizon_days": cfg["target"]["horizon"],
            "prob_up": p_last, "direction": "UP" if p_last >= thr else ("DOWN" if cfg["signal"]["mode"] == "long_short" else "STAY IN CASH"),
            "note": f"Uses only data up to {pd.Timestamp(last['date'].iloc[0]).strftime('%Y-%m-%d')} and the model trained on "
                    f"{split_info['train_start'].strftime('%Y-%m-%d')} → {split_info['train_end'].strftime('%Y-%m-%d')}. "
                    "A research signal with the test-period accuracy shown above, not investment advice.",
        }

    progress(0.60, "Walk-forward evaluation")
    if cfg["walk_forward"]["enabled"]:
        res["walk_forward"] = walk_forward(frame, cfg, progress)
        if res["walk_forward"].get("skipped"):
            res["walk_forward"].pop("summary", None)
    else:
        res["walk_forward"] = {"enabled": False}

    progress(0.82, "Regime analysis")
    res["regimes"] = regime_breakdown(df, test, bt_test["net"], bh_test["net"], p_test, train.index, cfg)

    gen = []
    for i, gid in enumerate(cfg["generalization_dataset_ids"]):
        progress(0.86 + 0.08 * i / max(1, len(cfg["generalization_dataset_ids"])), f"Testing on unseen asset #{gid}")
        try:
            gdf, gmeta = load_dataset(gid)
            gframe = prepare(gdf, cfg)
            gtest = gframe[(gframe.index >= split_info["test_start"]) & (gframe.index <= split_info["test_end"])]
            note = "same test dates as the training asset"
            if len(gtest) < 40:
                gtest = gframe.iloc[int(len(gframe) * 0.7):]
                note = "last 30% of this asset's history (test dates did not overlap)"
            gp = _proba_up(model, gtest[feats])
            gpm = predictive_metrics(gtest["y"].values, gp, thr)
            gbt = backtest(gtest["close"], pd.Series(gp, index=gtest.index), cfg)
            gbh = backtest(gtest["close"], None, cfg, buy_and_hold=True)
            gen.append({"dataset_id": gid, "name": gmeta.get("name"), "period": note,
                        "start": gtest.index[0], "end": gtest.index[-1], "days": len(gtest),
                        "accuracy": gpm["accuracy"], "roc_auc": gpm["roc_auc"], "edge_vs_majority": gpm["edge_vs_majority"],
                        "strategy_return": gbt["stats"]["total_return"], "strategy_sharpe": gbt["stats"]["sharpe"],
                        "buy_hold_return": gbh["stats"]["total_return"], "buy_hold_sharpe": gbh["stats"]["sharpe"]})
        except Exception as exc:  # one bad asset should not fail the experiment
            gen.append({"dataset_id": gid, "error": str(exc)})
    res["generalization"] = [g for g in gen if "error" not in g]
    res["generalization_errors"] = [g for g in gen if "error" in g]

    progress(0.96, "Diagnosing and scoring")
    res["diagnostics"] = diagnose(res, cfg)
    res["robustness"] = robustness_score(res, cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"])

    pred_fingerprint = hashlib.sha256(np.round(p_test, 10).tobytes()).hexdigest()
    metric_fingerprint = hashlib.sha256(json.dumps(clean_json({
        "test_acc": round(pm_test["accuracy"], 10), "test_sharpe": round(bt_test["stats"]["sharpe"], 10),
        "train_acc": round(pm_train["accuracy"], 10), "pred": pred_fingerprint,
    }), sort_keys=True).encode()).hexdigest()
    res["reproducibility"] = {
        "config_hash": config_hash(cfg),
        "dataset_sha256": ds_meta.get("sha256"),
        "seed": cfg["seed"],
        "result_fingerprint": metric_fingerprint,
        "prediction_fingerprint": pred_fingerprint,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    res["duration_seconds"] = round(time.time() - t_start, 2)
    progress(1.0, "Done")
    return clean_json(res)
