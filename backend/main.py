"""SignalForge API + static frontend. Run: uvicorn backend.main:app --reload"""
from __future__ import annotations

import asyncio
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import db
from .data import DatasetError, load_clean, parse_upload, preview_series, sha256_bytes, summarize, validate_csv
from .engine import (MODELS, ConfigError, build_portfolio, canonical_config, config_hash, normalize_config,
                     run_experiment)
from .features import DEFAULT_FEATURES, FEATURES, feature_catalog
from .report import build_report

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SAMPLE = ROOT / "sample_data"
MAX_UPLOAD_MB = 150

executor = ThreadPoolExecutor(max_workers=int(os.environ.get("SIGNALFORGE_WORKERS", "2")))
LIVE: dict[int, dict] = {}  # in-memory progress for running jobs (also persisted to SQLite)


# --------------------------------------------------------------------------- presets

def presets() -> list[dict]:
    return [
        {"key": "baseline_lr", "name": "Baseline: Logistic Regression",
         "description": "Simple, regularized linear model on 8 standard indicators. Walk-forward + tested on other assets.",
         "config": {"features": DEFAULT_FEATURES, "model": {"type": "logistic_regression", "params": {"C": 0.1}}}},
        {"key": "rf", "name": "Random Forest (regularized)",
         "description": "Shallow trees with large leaves, so it cannot memorise individual days.",
         "config": {"features": DEFAULT_FEATURES, "model": {"type": "random_forest", "params": {"max_depth": 4, "min_samples_leaf": 40}}}},
        {"key": "gb", "name": "Gradient Boosting",
         "description": "Boosted shallow trees with a small learning rate.",
         "config": {"features": DEFAULT_FEATURES, "model": {"type": "gradient_boosting", "params": {}}}},
        {"key": "overfit", "name": "Overfit Trap (demo)",
         "description": "Unlimited-depth tree, 24 features, 18 months of training data. ~100% train accuracy, coin-flip on unseen data.",
         "config": {"features": list(FEATURES), "model": {"type": "decision_tree", "params": {"max_depth": None, "min_samples_leaf": 1}},
                    "split": {"mode": "date", "train_start": "2019-01-01", "train_end": "2020-06-30"}}},
        {"key": "zero_cost", "name": "Same RF, zero costs (illusion)",
         "description": "Identical to the regularized Random Forest but with free trading. Compare to see how costs change the verdict.",
         "config": {"features": DEFAULT_FEATURES, "model": {"type": "random_forest", "params": {"max_depth": 4, "min_samples_leaf": 40}},
                    "costs": {"commission_bps": 0, "slippage_bps": 0}}},
        {"key": "leak", "name": "Leakage Demo (look-ahead)",
         "description": "Adds a centered moving average that peeks 5 days ahead. Looks brilliant, gets caught and disqualified.",
         "config": {"features": DEFAULT_FEATURES + ["sma_centered_LEAK"], "allow_lookahead": True,
                    "model": {"type": "logistic_regression", "params": {"C": 0.1}}}},
    ]


# --------------------------------------------------------------------------- dataset helpers

def _dataset_row(did: int) -> dict:
    row = db.loads(db.one("SELECT * FROM datasets WHERE id=?", (did,)), "validation")
    if not row:
        raise HTTPException(404, f"Dataset #{did} not found")
    return row


def _load_for_engine(did: int):
    row = _dataset_row(did)
    if not Path(row["path"]).exists():
        raise ConfigError(f"Data file for dataset #{did} is missing on disk.")
    raw = Path(row["path"]).read_bytes()
    if sha256_bytes(raw) != row["sha256"]:
        raise ConfigError(f"Dataset #{did} has changed on disk since upload (checksum mismatch); results would not be reproducible.")
    return load_clean(row["path"]), row


def _save_dataset(project_id: int, name: str, filename: str, raw: bytes) -> dict:
    df, report = validate_csv(raw, filename)
    return _store_dataset(project_id, name, filename, df, report)


def _store_dataset(project_id: int, name: str, filename: str, df: pd.DataFrame, report: dict) -> dict:
    did = db.execute(
        "INSERT INTO datasets(project_id,name,filename,path,sha256,rows,start_date,end_date,has_volume,validation_json,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (project_id, name, filename, "", "", len(df), report["start"], report["end"], int(df["volume"].notna().any()),
         json.dumps(report), db.now()))
    path = db.DATA_DIR / "datasets" / f"{did}.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    db.execute("UPDATE datasets SET path=?, sha256=? WHERE id=?", (str(path), sha256_bytes(path.read_bytes()), did))
    return _dataset_row(did)


# --------------------------------------------------------------------------- jobs

def _progress(eid: int):
    def cb(p: float, stage: str):
        LIVE[eid] = {"status": "running", "progress": round(p, 3), "stage": stage}
        db.execute("UPDATE experiments SET progress=?, stage=? WHERE id=?", (p, stage, eid))
    return cb


def _run_job(eid: int):
    exp = db.loads(db.one("SELECT * FROM experiments WHERE id=?", (eid,)), "config")
    db.execute("UPDATE experiments SET status='running', started_at=? WHERE id=?", (db.now(), eid))
    LIVE[eid] = {"status": "running", "progress": 0.0, "stage": "Starting"}
    try:
        results = run_experiment(exp["config"], _load_for_engine, _progress(eid))
        if exp.get("reproduce_of"):
            orig = db.loads(db.one("SELECT * FROM experiments WHERE id=?", (exp["reproduce_of"],)), "results")
            if orig and orig.get("results"):
                o, n = orig["results"]["reproducibility"], results["reproducibility"]
                checks = {
                    "config_hash_match": o["config_hash"] == n["config_hash"],
                    "dataset_match": o["dataset_sha256"] == n["dataset_sha256"],
                    "predictions_match": o["prediction_fingerprint"] == n["prediction_fingerprint"],
                    "metrics_match": o["result_fingerprint"] == n["result_fingerprint"],
                }
                results["reproduction_check"] = {
                    "original_id": exp["reproduce_of"], **checks, "reproduced": all(checks.values()),
                    "original_test_sharpe": orig["results"]["strategy"]["test"]["sharpe"],
                    "new_test_sharpe": results["strategy"]["test"]["sharpe"],
                    "environment_original": o["environment"], "environment_now": n["environment"],
                }
        score = results["robustness"]["score"]
        db.execute("UPDATE experiments SET status='done', progress=1, stage='Done', results_json=?, score=?, finished_at=? WHERE id=?",
                   (json.dumps(results), score, db.now(), eid))
        LIVE[eid] = {"status": "done", "progress": 1.0, "stage": "Done"}
    except ConfigError as exc:
        db.execute("UPDATE experiments SET status='failed', error=?, finished_at=? WHERE id=?", (str(exc), db.now(), eid))
        LIVE[eid] = {"status": "failed", "progress": 1.0, "stage": "Failed", "error": str(exc)}
    except Exception as exc:  # unexpected: keep the trace in the log, show a clean message
        traceback.print_exc()
        msg = f"Unexpected error: {type(exc).__name__}: {exc}"
        db.execute("UPDATE experiments SET status='failed', error=?, finished_at=? WHERE id=?", (msg, db.now(), eid))
        LIVE[eid] = {"status": "failed", "progress": 1.0, "stage": "Failed", "error": msg}


def _diff(a: dict, b: dict, prefix="") -> list[dict]:
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        key = f"{prefix}{k}"
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _diff(va, vb, key + ".")
        elif va != vb:
            out.append({"field": key, "before": va, "after": vb})
    return out


def create_experiment(project_id: int, body: dict) -> dict:
    datasets = db.query("SELECT id, has_volume FROM datasets WHERE project_id=?", (project_id,))
    ids = {d["id"] for d in datasets}
    vol = {d["id"]: bool(d["has_volume"]) for d in datasets}
    user_cfg = body.get("config", body)
    try:
        cfg = normalize_config(user_cfg, ids, vol.get(int(user_cfg.get("dataset_id") or -1), True))
    except ConfigError as exc:
        raise HTTPException(422, str(exc))
    bad_gen = [g for g in cfg["generalization_dataset_ids"] if g not in ids]
    if bad_gen:
        raise HTTPException(422, f"Generalization dataset(s) {bad_gen} are not in this project.")
    name = (body.get("name") or user_cfg.get("name") or "").strip() or f"{MODELS[cfg['model']['type']]['label']} · {len(cfg['features'])} features"
    cfg["name"] = name
    parent_id = body.get("parent_id")
    version, diff = 1, None
    if parent_id:
        parent = db.loads(db.one("SELECT * FROM experiments WHERE id=? AND project_id=?", (parent_id, project_id)), "config")
        if not parent:
            raise HTTPException(404, "Parent experiment not found")
        root = parent
        version = (db.one("SELECT MAX(version) v FROM experiments WHERE id=? OR parent_id=?", (parent_id, parent_id))["v"] or 1) + 1
        diff = _diff(canonical_config(root["config"]), canonical_config(cfg))
    eid = db.execute(
        "INSERT INTO experiments(project_id,name,notes,parent_id,version,reproduce_of,config_json,config_hash,config_diff_json,status,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,'queued',?)",
        (project_id, name, body.get("notes", ""), parent_id, version, body.get("reproduce_of"), json.dumps(cfg),
         config_hash(cfg), json.dumps(diff) if diff is not None else None, db.now()))
    LIVE[eid] = {"status": "queued", "progress": 0.0, "stage": "Queued"}
    executor.submit(_run_job, eid)
    return _exp_summary(db.one("SELECT * FROM experiments WHERE id=?", (eid,)))


def _exp_summary(row: dict) -> dict:
    cfg = json.loads(row["config_json"])
    out = {k: row[k] for k in ("id", "project_id", "name", "notes", "parent_id", "version", "reproduce_of", "config_hash",
                               "status", "progress", "stage", "error", "score", "created_at", "started_at", "finished_at")}
    out["model"] = MODELS[cfg["model"]["type"]]["label"]
    out["dataset_id"] = cfg["dataset_id"]
    out["n_features"] = len(cfg["features"])
    if row.get("results_json"):
        r = json.loads(row["results_json"])
        out["headline"] = {
            "train_accuracy": r["predictive"]["train"]["accuracy"], "test_accuracy": r["predictive"]["test"]["accuracy"],
            "test_auc": r["predictive"]["test"]["roc_auc"], "test_return": r["strategy"]["test"]["total_return"],
            "test_sharpe": r["strategy"]["test"]["sharpe"], "test_max_dd": r["strategy"]["test"]["max_drawdown"],
            "buy_hold_sharpe": r["benchmark"]["test"]["sharpe"], "buy_hold_return": r["benchmark"]["test"]["total_return"],
            "wf_mean_sharpe": (r.get("walk_forward") or {}).get("summary", {}).get("mean_sharpe"),
            "disqualified": r["robustness"]["disqualified"], "dataset": r["dataset"]["name"],
            "disqualified_reasons": r["robustness"].get("disqualified_reasons", []),
            "raw_score": r["robustness"].get("raw_score"),
            "reproduced": (r.get("reproduction_check") or {}).get("reproduced"),
        }
    if row["id"] in LIVE and row["status"] in ("queued", "running"):
        out.update({k: LIVE[row["id"]][k] for k in ("progress", "stage")})
    return out


# --------------------------------------------------------------------------- demo seed

def seed_demo(run_experiments: bool = True) -> int:
    pid = db.execute("INSERT INTO projects(name, description, created_at) VALUES(?,?,?)",
                     ("Demo: Signal research (synthetic NSE-style data)",
                      "Four synthetic assets with bull/bear regimes and a weak momentum effect, plus a messy file to show validation.",
                      db.now()))
    ids = {}
    for f in sorted(SAMPLE.glob("*.csv")) + [SAMPLE / "bad" / "MESSY_but_usable.csv"]:
        if f.exists():
            ids[f.stem] = _save_dataset(pid, f.stem, f.name, f.read_bytes())["id"]
    if run_experiments and "DEMO_TECH" in ids:
        others = [ids[k] for k in ("DEMO_BANK", "DEMO_PHARMA", "DEMO_INDEX") if k in ids]
        for p in presets():
            cfg = {**p["config"], "dataset_id": ids["DEMO_TECH"], "generalization_dataset_ids": others}
            create_experiment(pid, {"name": p["name"], "config": cfg, "notes": p["description"]})
        _seed_per_stock_baselines(pid, {k.upper(): v for k, v in ids.items()}, ["DEMO_BANK", "DEMO_PHARMA", "DEMO_INDEX"])
    return pid


REAL = SAMPLE / "real"
REAL_PREFERRED = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "NIFTY50", "NSEI"]


def seed_real(run_experiments: bool = True) -> int | None:
    """Project built from real NSE files placed in sample_data/real/ (CSV, Parquet or ZIP)."""
    files = sorted(p for p in REAL.glob("*") if p.suffix.lower() in (".csv", ".parquet", ".pq", ".zip")) if REAL.exists() else []
    if not files:
        return None
    pid = db.execute("INSERT INTO projects(name, description, created_at) VALUES(?,?,?)",
                     ("NSE stocks (real data)",
                      "Real daily prices for NIFTY-50 constituents, split/bonus-adjusted on import. "
                      "Experiments train on one stock and are tested on the others.", db.now()))
    ids = {}
    for f in files:
        for it in parse_upload(f.read_bytes(), f.name):
            if "df" in it and it["name"].upper() not in ids:
                ids[it["name"].upper()] = _store_dataset(pid, it["name"], f.name, it["df"], it["report"])["id"]
    if run_experiments and ids:
        main_sym = next((s for s in REAL_PREFERRED if s in ids), sorted(ids)[0])
        gen_order = ["HDFCBANK", "INFY", "TCS", "ICICIBANK", "ITC", "SBIN"] + sorted(ids)
        others = list(dict.fromkeys(ids[k] for k in gen_order if k in ids and k != main_sym))[:6]
        for p in presets():
            cfg = {**p["config"], "dataset_id": ids[main_sym], "generalization_dataset_ids": others}
            create_experiment(pid, {"name": p["name"], "config": cfg, "notes": p["description"]})
        _seed_per_stock_baselines(pid, ids, [k for k in gen_order if k in ids and k != main_sym][:4])
    return pid


def _seed_per_stock_baselines(pid: int, ids: dict, symbols: list[str]):
    """The baseline on a few more stocks, so 'Compare' and 'Portfolio' have one signal per stock to combine."""
    base = presets()[0]
    for sym in symbols:
        cfg = {**base["config"], "dataset_id": ids[sym], "walk_forward": {"enabled": True}}
        create_experiment(pid, {"name": f"Baseline LR · {sym}", "config": cfg,
                                "notes": "Same baseline settings on another stock; combine these in a portfolio on the Compare page."})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Requeue jobs interrupted by a restart; seed the demo workspace on first boot.
    for row in db.query("SELECT id FROM experiments WHERE status IN ('queued','running')"):
        executor.submit(_run_job, row["id"])
    if not db.one("SELECT id FROM projects LIMIT 1") and os.environ.get("SIGNALFORGE_SEED", "1") == "1":
        seed_demo()
        seed_real()  # created last, so it opens first when real data is bundled
    yield


app = FastAPI(title="SignalForge", version="1.0", lifespan=lifespan)


@app.exception_handler(DatasetError)
async def dataset_error_handler(_, exc: DatasetError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "report": exc.report})


# --------------------------------------------------------------------------- routes: meta

@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    return {"features": feature_catalog(), "default_features": DEFAULT_FEATURES, "models": MODELS, "presets": presets()}


# --------------------------------------------------------------------------- routes: projects

@app.get("/api/projects")
def list_projects():
    return db.query("""SELECT p.*, (SELECT COUNT(*) FROM datasets d WHERE d.project_id=p.id) n_datasets,
                       (SELECT COUNT(*) FROM experiments e WHERE e.project_id=p.id) n_experiments
                       FROM projects p ORDER BY p.id DESC""")


@app.post("/api/projects")
def create_project(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "Project name is required.")
    if len(name) > 120:
        raise HTTPException(422, "Project name must be 120 characters or fewer.")
    pid = db.execute("INSERT INTO projects(name, description, created_at) VALUES(?,?,?)",
                     (name, (body.get("description") or "").strip(), db.now()))
    return db.one("SELECT * FROM projects WHERE id=?", (pid,))


@app.get("/api/projects/{pid}")
def get_project(pid: int):
    p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@app.patch("/api/projects/{pid}")
def update_project(pid: int, body: dict):
    get_project(pid)
    if "name" in body and not str(body["name"]).strip():
        raise HTTPException(422, "Project name cannot be empty.")
    for k in ("name", "description"):
        if k in body:
            db.execute(f"UPDATE projects SET {k}=? WHERE id=?", (str(body[k]).strip(), pid))
    return get_project(pid)


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    get_project(pid)
    for d in db.query("SELECT path FROM datasets WHERE project_id=?", (pid,)):
        Path(d["path"]).unlink(missing_ok=True)
    db.execute("DELETE FROM projects WHERE id=?", (pid,))
    return {"deleted": pid}


@app.post("/api/demo/seed")
def api_seed():
    return {"project_id": seed_demo()}


# --------------------------------------------------------------------------- routes: datasets

@app.get("/api/projects/{pid}/datasets")
def list_datasets(pid: int):
    get_project(pid)
    rows = db.query("SELECT * FROM datasets WHERE project_id=? ORDER BY id", (pid,))
    return [db.loads(r, "validation") for r in rows]


@app.post("/api/projects/{pid}/datasets")
async def upload_dataset(pid: int, file: UploadFile = File(...), name: str = Form("")):
    get_project(pid)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB. Upload fewer stocks, or a ZIP.")
    fname = file.filename or "upload.csv"
    items = await run_in_threadpool(parse_upload, raw, fname)
    existing = {r["name"].upper() for r in db.query("SELECT name FROM datasets WHERE project_id=?", (pid,))}
    ok = [i for i in items if "df" in i]
    created, skipped = [], []
    for it in items:
        if "df" not in it:
            skipped.append({"name": it["name"], "reason": it["error"], "report": it["report"], "ignored": bool(it.get("skipped"))})
            continue
        nm = ((name.strip() if name and len(ok) == 1 else "") or it["name"]).strip()[:80]
        if nm.upper() in existing:
            skipped.append({"name": nm, "reason": f"A dataset named {nm} already exists in this project. Delete it first to re-import.",
                            "report": it["report"], "ignored": True})
            continue
        existing.add(nm.upper())
        created.append(_store_dataset(pid, nm, PurePosixPath(fname).name if len(ok) == 1 else f"{PurePosixPath(fname).name} → {it['report']['filename']}",
                                      it["df"], it["report"]))
    if not created:
        real = [s for s in skipped if not s["ignored"]]
        dupes = [s for s in skipped if "already exists" in s["reason"]]
        if not real and dupes:
            detail, report = f"Nothing new to import: {len(dupes)} stock(s) from this file are already in the project.", None
        elif real:
            detail, report = real[0]["reason"], real[0]["report"]
        else:
            detail = skipped[0]["reason"] if skipped else "Nothing could be imported from this file."
            report = skipped[0]["report"] if skipped else None
        return JSONResponse(status_code=422, content={"detail": detail, "report": report, "skipped": skipped})
    return {"created": created, "skipped": skipped}


@app.get("/api/datasets/{did}")
def get_dataset(did: int):
    row = _dataset_row(did)
    df = load_clean(row["path"])
    row["summary"] = summarize(df)
    row["series"] = preview_series(df)
    head = df.head(8).copy()
    head["date"] = head["date"].dt.strftime("%Y-%m-%d")
    row["head"] = head.round(4).astype(object).where(head.notna(), None).to_dict(orient="records")
    row["used_by"] = [r["id"] for r in db.query("SELECT id, config_json FROM experiments") if json.loads(r["config_json"]).get("dataset_id") == did]
    return row


@app.delete("/api/datasets/{did}")
def delete_dataset(did: int):
    row = _dataset_row(did)
    used = [r["id"] for r in db.query("SELECT id, config_json FROM experiments WHERE project_id=?", (row["project_id"],))
            if did == json.loads(r["config_json"]).get("dataset_id") or did in json.loads(r["config_json"]).get("generalization_dataset_ids", [])]
    if used:
        raise HTTPException(409, f"Dataset is used by experiment(s) {used}. Deleting it would make them impossible to reproduce.")
    Path(row["path"]).unlink(missing_ok=True)
    db.execute("DELETE FROM datasets WHERE id=?", (did,))
    return {"deleted": did}


# --------------------------------------------------------------------------- routes: experiments

@app.get("/api/projects/{pid}/experiments")
def list_experiments(pid: int):
    get_project(pid)
    return [_exp_summary(r) for r in db.query("SELECT * FROM experiments WHERE project_id=? ORDER BY id DESC", (pid,))]


@app.post("/api/projects/{pid}/experiments")
def post_experiment(pid: int, body: dict):
    get_project(pid)
    return create_experiment(pid, body)


@app.get("/api/experiments/{eid}")
def get_experiment(eid: int):
    row = db.one("SELECT * FROM experiments WHERE id=?", (eid,))
    if not row:
        raise HTTPException(404, "Experiment not found")
    out = _exp_summary(row)
    out["config"] = json.loads(row["config_json"])
    out["config_diff"] = json.loads(row["config_diff_json"]) if row["config_diff_json"] else None
    out["results"] = json.loads(row["results_json"]) if row["results_json"] else None
    out["lineage"] = [_exp_summary(r) for r in db.query(
        "SELECT * FROM experiments WHERE id=? OR parent_id=? OR id=? ORDER BY version",
        (eid, row["parent_id"] or eid, row["parent_id"] or -1))]
    return out


@app.patch("/api/experiments/{eid}")
def update_experiment(eid: int, body: dict):
    get_experiment(eid)
    for k in ("name", "notes"):
        if k in body:
            db.execute(f"UPDATE experiments SET {k}=? WHERE id=?", (str(body[k]).strip(), eid))
    return get_experiment(eid)


@app.delete("/api/experiments/{eid}")
def delete_experiment(eid: int):
    exp = get_experiment(eid)
    if exp["status"] in ("queued", "running"):
        raise HTTPException(409, "Cannot delete an experiment while it is running.")
    db.execute("DELETE FROM experiments WHERE id=?", (eid,))
    return {"deleted": eid}


@app.post("/api/experiments/{eid}/reproduce")
def reproduce(eid: int):
    exp = get_experiment(eid)
    if exp["status"] != "done":
        raise HTTPException(409, "Only finished experiments can be reproduced.")
    body = {"name": f"Reproduction of #{eid}: {exp['name']}", "config": exp["config"], "reproduce_of": eid,
            "notes": "Re-run from the saved configuration to verify identical results."}
    return create_experiment(exp["project_id"], body)


@app.get("/api/experiments/{eid}/events")
async def experiment_events(eid: int):
    get_experiment(eid)

    async def stream():
        last = None
        for _ in range(3600):  # up to ~30 minutes
            state = LIVE.get(eid)
            if state is None:
                row = db.one("SELECT status, progress, stage, error FROM experiments WHERE id=?", (eid,))
                state = dict(row) if row else {"status": "failed", "error": "deleted"}
            payload = json.dumps(state)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if state["status"] in ("done", "failed"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/experiments/{eid}/report", response_class=HTMLResponse)
def report(eid: int, download: int = 0):
    exp = get_experiment(eid)
    if exp["status"] != "done":
        raise HTTPException(409, "Report is available once the experiment has finished.")
    page = build_report(exp, db.one("SELECT * FROM projects WHERE id=?", (exp["project_id"],)))
    headers = {"Content-Disposition": f'attachment; filename="signalforge_experiment_{eid}.html"'} if download else {}
    return HTMLResponse(page, headers=headers)


@app.get("/api/projects/{pid}/leaderboard")
def leaderboard(pid: int):
    rows = [e for e in list_experiments(pid) if e["status"] == "done"]
    ranked = sorted([e for e in rows if e["score"] is not None], key=lambda e: -e["score"])
    disq = [e for e in rows if e["score"] is None]
    by_train = sorted(rows, key=lambda e: -(e.get("headline") or {}).get("train_accuracy", 0))
    return {"ranked": ranked, "disqualified": disq,
            "naive_rank_by_train_accuracy": [e["id"] for e in by_train],
            "formula": "OOS Sharpe − 0.5×Sharpe std across walk-forward folds (or Sharpe standard error) − 3×(train acc − test acc) + 0.5×(Sharpe − buy&hold Sharpe, capped ±1). "
                       "Disqualified: look-ahead leakage, or costs below 5 bps."}


@app.post("/api/projects/{pid}/portfolio")
def portfolio(pid: int, body: dict):
    get_project(pid)
    try:
        ids = [int(x) for x in body.get("experiment_ids", [])][:12]
        max_weight = float(body.get("max_weight", 0.25))
        max_gross = float(body.get("max_gross", 1.0))
        cost_bps = float(body.get("commission_bps", 10)) + float(body.get("slippage_bps", 5))
    except (TypeError, ValueError):
        raise HTTPException(422, "Weights and costs must be numbers.")
    legs = []
    for i in ids:
        e = get_experiment(i)
        if e["project_id"] != pid:
            raise HTTPException(422, f"Experiment #{i} belongs to another project.")
        if e["status"] != "done":
            raise HTTPException(422, f"Experiment #{i} has not finished.")
        r = e["results"]
        if r["robustness"]["disqualified"] and any(v["leaky"] for v in r["leakage"]["features"].values()):
            raise HTTPException(422, f"Experiment #{i} ({e['name']}) uses look-ahead features, so it cannot be traded in a portfolio.")
        if not r.get("daily_test"):
            raise HTTPException(422, f"Experiment #{i} was run before portfolios existed. Click Reproduce on it, then use the new run.")
        legs.append({"id": i, "name": r["dataset"]["name"], "daily": r["daily_test"]})
    try:
        return build_portfolio(legs, max_weight, max_gross, cost_bps)
    except ConfigError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/compare")
def compare(ids: str):
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()][:6]
    except ValueError:
        raise HTTPException(422, "ids must be a comma-separated list of experiment ids")
    out = []
    for i in id_list:
        e = get_experiment(i)
        if e["status"] == "done":
            out.append(e)
    if len(out) < 2:
        raise HTTPException(422, "Pick at least two finished experiments to compare.")
    for e in out[1:]:
        e["config_diff_vs_first"] = _diff(canonical_config(out[0]["config"]), canonical_config(e["config"]))
    return out


# --------------------------------------------------------------------------- frontend

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/sample/{name}")
def sample_file(name: str):
    allowed = {p.name: p for p in list(SAMPLE.glob("*.csv")) + list((SAMPLE / "bad").glob("*.csv"))}
    if name not in allowed:
        raise HTTPException(404, "Sample not found")
    return FileResponse(allowed[name], filename=name, media_type="text/csv")


@app.get("/api/samples")
def samples():
    return [{"name": p.name, "kind": "bad" if p.parent.name == "bad" else "good"}
            for p in sorted(SAMPLE.glob("*.csv")) + sorted((SAMPLE / "bad").glob("*.csv"))]
