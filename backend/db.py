"""SQLite persistence (single file, zero setup). Thread-safe via one lock."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("SIGNALFORGE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "datasets").mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "signalforge.db"

_lock = threading.RLock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA foreign_keys=ON")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    filename TEXT,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    rows INTEGER,
    start_date TEXT,
    end_date TEXT,
    has_volume INTEGER DEFAULT 1,
    validation_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    notes TEXT DEFAULT '',
    parent_id INTEGER,
    version INTEGER DEFAULT 1,
    reproduce_of INTEGER,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_diff_json TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL DEFAULT 0,
    stage TEXT DEFAULT '',
    error TEXT,
    results_json TEXT,
    score REAL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id, status);
CREATE INDEX IF NOT EXISTS idx_experiments_parent ON experiments(parent_id);
"""
with _lock:
    _conn.executescript(SCHEMA)
    _conn.commit()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def execute(sql: str, params: tuple = ()) -> int:
    with _lock:
        cur = _conn.execute(sql, params)
        _conn.commit()
        return cur.lastrowid


def query(sql: str, params: tuple = ()) -> list[dict]:
    with _lock:
        return [dict(r) for r in _conn.execute(sql, params).fetchall()]


def one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def loads(row: dict | None, *keys: str) -> dict | None:
    if row is None:
        return None
    for k in keys:
        col = f"{k}_json"
        if col in row:
            row[k] = json.loads(row[col]) if row[col] else None
            del row[col]
    return row
