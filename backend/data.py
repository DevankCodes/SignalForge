"""Dataset import, validation and cleaning.

Accepted uploads
  * CSV or Parquet with one asset (date + close required; open/high/low/volume optional)
  * a file with many symbols (e.g. Kaggle's NIFTY50_all.csv): split into one dataset per symbol
  * a file whose symbol was renamed over time (e.g. INFOSYSTCH -> INFY): merged into one series
  * a ZIP of any of the above (e.g. Kaggle's archive.zip): every member is imported

Cleaning never leaks the future: gaps are forward-filled only (past -> present), never back-filled.
Corporate actions (splits/bonus issues) are detected and earlier prices are adjusted, because an
unadjusted 1:2 split looks like a 50% crash and would poison returns, features and backtests.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

MIN_ROWS = 250  # ~1 trading year; needed for 200-day features + a test period
MAX_MEMBERS = 120

CANONICAL = {"date", "open", "high", "low", "close", "adj_close", "volume", "prev_close", "symbol", "series"}
COLUMN_ALIASES = {
    "timestamp": "date", "datetime": "date", "time": "date", "day": "date", "trade_date": "date",
    "close_price": "close", "closing_price": "close", "price": "close", "last": "close", "ltp": "close",
    "adjclose": "adj_close", "adjusted_close": "adj_close", "adj_close_price": "adj_close",
    "vol": "volume", "shares_traded": "volume", "total_traded_quantity": "volume", "qty": "volume",
    "prevclose": "prev_close", "previous_close": "prev_close", "prev_close_price": "prev_close",
    "ticker": "symbol", "tradingsymbol": "symbol", "scrip": "symbol", "stock": "symbol",
}
PRICE_COLS = ["open", "high", "low", "close"]
SPLIT_CANDIDATES = {0.5: "1:2 split or 1:1 bonus", 1 / 3: "1:3 split or 2:1 bonus", 0.25: "1:4 split or 3:1 bonus",
                    0.2: "1:5 split or 4:1 bonus", 0.1: "1:10 split", 2 / 3: "1:2 bonus", 0.4: "2:5 split"}


class DatasetError(Exception):
    """Raised when a file cannot be used at all. Carries the validation report."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "").replace("%", "pct_")


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Exact names win over aliases (Kaggle NSE files have both 'Last' and 'Close')."""
    rename, taken = {}, set()
    for col in df.columns:
        key = _norm(col)
        if key in CANONICAL and key not in taken:
            rename[col] = key
            taken.add(key)
    for col in df.columns:
        if col in rename:
            continue
        target = COLUMN_ALIASES.get(_norm(col))
        if target and target not in taken:
            rename[col] = target
            taken.add(target)
    return df.rename(columns=rename)


# --------------------------------------------------------------------------- reading

def _read_table(raw: bytes, filename: str) -> pd.DataFrame:
    low = filename.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(raw))
    if low.endswith((".xlsx", ".xls")):
        raise ValueError("Excel files are not supported; export the sheet as CSV.")
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def _split_symbols(df: pd.DataFrame, filename: str) -> list[tuple[str, pd.DataFrame, list[str]]]:
    """Return [(name, frame, notes)]. Handles single, renamed and multi-symbol tables."""
    stem = PurePosixPath(filename).stem
    if "symbol" not in df.columns:
        return [(stem, df, [])]
    sym = df["symbol"].astype(str).str.strip()
    df = df.assign(symbol=sym)
    syms = [s for s in sym.dropna().unique() if s and s.lower() != "nan"]
    if len(syms) <= 1:
        return [(syms[0] if syms else stem, df, [])]
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed") if "date" in df.columns else None
    if dates is None:
        return [(stem, df, [])]
    spans = sorted(((dates[sym == s].min(), dates[sym == s].max(), s) for s in syms), key=lambda x: x[0])
    sequential = all(spans[i + 1][0] >= spans[i][1] - pd.Timedelta(days=5) for i in range(len(spans) - 1))
    if sequential:  # one company that changed its ticker over time
        chain = " → ".join(s for _, _, s in spans)
        return [(spans[-1][2], df, [f"Symbol changed over time ({chain}); merged into one continuous series."])]

    # Multi-symbol file. Link renamed tickers: B starts within a week of A ending, at a continuous price.
    span = {s: (a, b) for a, b, s in spans}
    close = pd.to_numeric(df["close"] if "close" in df.columns else df.get("adj_close"), errors="coerce")
    ref = pd.to_numeric(df["prev_close"], errors="coerce") if "prev_close" in df.columns else close

    def last_px(s):
        m = (sym == s) & (dates == span[s][1])
        return close[m].iloc[-1] if m.any() else np.nan

    def first_px(s):
        m = (sym == s) & (dates == span[s][0])
        return ref[m].iloc[0] if m.any() else np.nan

    successor = {}
    for a in syms:
        cands = [b for b in syms if b != a and b not in successor.values()
                 and pd.Timedelta(0) < span[b][0] - span[a][1] <= pd.Timedelta(days=7)
                 and abs(first_px(b) / last_px(a) - 1) < 0.1]
        if len(cands) == 1:
            successor[a] = cands[0]
    heads = [s for s in syms if s not in successor.values()]
    out = []
    for h in heads:
        chain = [h]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        notes = [f"Imported from multi-symbol file {PurePosixPath(filename).name}."]
        if len(chain) > 1:
            notes.append(f"Symbol changed over time ({' → '.join(chain)}); merged into one continuous series.")
        out.append((chain[-1], df[sym.isin(chain)], notes))
    return out


def _clean_symbol(name: str) -> str:
    """'HDFCBANK.NS' -> 'HDFCBANK', '^NSEI' -> 'NSEI' (Yahoo-style tickers from the Hugging Face dataset)."""
    n = str(name).strip()
    for suffix in (".NS", ".BO", ".ns", ".bo"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.lstrip("^") or str(name)


def parse_upload(raw: bytes, filename: str) -> list[dict]:
    """Turn one upload into a list of results: {"name", "df", "report"} or {"name", "error", "report"}."""
    results: list[dict] = []
    if not raw or not raw.strip():
        return [{"name": PurePosixPath(filename).stem, "error": "The file is empty.",
                 "report": {"filename": filename, "errors": ["The file is empty."], "warnings": [], "fixes": []}}]

    tables: list[tuple[str, bytes]] = []
    if filename.lower().endswith(".zip") or raw[:2] == b"PK" and not filename.lower().endswith((".parquet", ".pq", ".xlsx")):
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            return [{"name": filename, "error": "The ZIP file is corrupted.",
                     "report": {"filename": filename, "errors": ["The ZIP file is corrupted or incomplete."], "warnings": [], "fixes": []}}]
        members = [m for m in zf.infolist() if not m.is_dir() and not PurePosixPath(m.filename).name.startswith((".", "_"))
                   and m.filename.lower().endswith((".csv", ".txt", ".parquet", ".pq"))]
        # Smallest first: per-stock files are imported before a combined file that repeats them.
        for m in sorted(members, key=lambda m: m.file_size)[:MAX_MEMBERS]:
            tables.append((m.filename, zf.read(m)))
        if not tables:
            return [{"name": filename, "error": "No CSV or Parquet files inside the ZIP.",
                     "report": {"filename": filename, "errors": ["The ZIP contains no .csv or .parquet files."], "warnings": [], "fixes": []}}]
    else:
        tables.append((filename, raw))

    seen: set[str] = set()
    for fname, data in tables:
        short = PurePosixPath(fname).name
        try:
            table = _map_columns(_read_table(data, fname))
        except Exception as exc:
            results.append({"name": PurePosixPath(fname).stem, "error": f"Could not read {short}: {exc}",
                            "report": {"filename": short, "errors": [f"Could not read the file: {exc}"], "warnings": [], "fixes": []}})
            continue
        if "date" not in table.columns or ("close" not in table.columns and "adj_close" not in table.columns):
            results.append({"name": PurePosixPath(fname).stem, "skipped": True,
                            "error": f"{short} is not price data (no date/close columns); skipped.",
                            "report": {"filename": short, "errors": [
                                "Missing required column(s): " + ", ".join(c for c in ("date", "close") if c not in table.columns)
                                + ". Expected at least 'date' and 'close' (plus optional open, high, low, volume). "
                                + f"Found: {[str(c) for c in table.columns][:15]}"], "warnings": [], "fixes": []}})
            continue
        for name, frame, notes in _split_symbols(table, fname):
            name = _clean_symbol(name)
            key = name.upper()
            if key in seen:
                results.append({"name": name, "skipped": True, "error": f"{name} already imported from another file in this upload; duplicate skipped.",
                                "report": {"filename": short, "errors": [], "warnings": [], "fixes": []}})
                continue
            try:
                df, report = clean_frame(frame, short)
                report["fixes"][:0] = notes
                report["symbol"] = name
                results.append({"name": name, "df": df, "report": report})
                seen.add(key)
            except DatasetError as exc:
                exc.report["fixes"][:0] = notes
                results.append({"name": name, "error": str(exc), "report": exc.report})
    return results


def validate_csv(raw: bytes, filename: str = "upload.csv") -> tuple[pd.DataFrame, dict]:
    """Single-asset helper: returns (clean_df, report) or raises DatasetError."""
    items = parse_upload(raw, filename)
    ok = [i for i in items if "df" in i]
    if len(ok) == 1 and len(items) == 1:
        return ok[0]["df"], ok[0]["report"]
    if not ok:
        raise DatasetError(items[0]["error"], items[0]["report"])
    raise DatasetError("File contains more than one asset", {"filename": filename, "errors": [
        f"File contains {len(ok)} assets; use the multi-asset import."], "warnings": [], "fixes": []})


# --------------------------------------------------------------------------- cleaning

def _adjust_corporate_actions(df: pd.DataFrame, fixes: list, warnings: list) -> pd.DataFrame:
    """Back-adjust prices for splits and bonus issues.

    With an exchange 'prev_close' column (NSE bhavcopy), the exchange itself states the adjusted previous
    close on the ex-date, so factor = prev_close[t] / close[t-1] is exact. Without it, a large overnight gap
    that matches a standard split ratio is treated as a likely split.
    """
    close_prev = df["close"].shift(1)
    consecutive = (df["date"].diff().dt.days <= 5).values  # never infer across a data gap
    ref = df["open"] if "open" in df.columns and df["open"].notna().mean() > 0.9 else df["close"]
    gap = (ref / close_prev).values
    pc = pd.to_numeric(df["prev_close"], errors="coerce").values if "prev_close" in df.columns else np.full(len(df), np.nan)
    events = []
    for i in range(1, len(df)):
        if not consecutive[i] or not np.isfinite(gap[i]):
            continue
        exch = pc[i] / close_prev.iloc[i] if np.isfinite(pc[i]) else np.nan
        if np.isfinite(exch) and exch < 0.9 and abs(gap[i] / exch - 1) < 0.2:
            # The exchange published an adjusted previous close: exact factor.
            events.append((i, float(exch), "exchange-adjusted Prev Close"))
        elif gap[i] < 0.67:
            # A >33% overnight drop between consecutive sessions in a large-cap is almost always a split/bonus.
            best = min(SPLIT_CANDIDATES, key=lambda c: abs(gap[i] / c - 1))
            if abs(gap[i] / best - 1) < 0.05:
                events.append((i, best, SPLIT_CANDIDATES[best]))
            else:
                events.append((i, float(gap[i]), "ratio taken from the overnight gap"))
    if not events:
        return df
    mult = np.ones(len(df))
    for i, f, _ in events:
        mult[:i] *= f
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col] * mult
    if "volume" in df.columns:
        df["volume"] = df["volume"] / mult
    desc = ", ".join(f"{df['date'].iloc[i].strftime('%Y-%m-%d')} ×{f:.3g} ({src})" for i, f, src in events[:8])
    # (factor < 1 means earlier prices are scaled down to today's share count)
    more = f" and {len(events) - 8} more" if len(events) > 8 else ""
    fixes.append(f"Adjusted earlier prices for {len(events)} split/bonus event(s): {desc}{more}. "
                 "Without this, a split looks like a crash and corrupts returns.")
    if any("exchange" not in src for _, _, src in events):
        fixes.append("Split/bonus events were inferred from overnight price gaps (the file's Prev Close is not adjusted). "
                     "Verify against the company's corporate-action history if precision matters.")
    return df


def clean_frame(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, dict]:
    errors: list[str] = []
    warnings: list[str] = []
    fixes: list[str] = []
    report = {"filename": filename, "errors": errors, "warnings": warnings, "fixes": fixes}
    df = df.copy()
    report["rows_raw"] = int(len(df))
    report["columns_raw"] = [str(c) for c in df.columns]

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
        fixes.append("No 'close' column found; using 'adj_close' as close.")
    missing_required = [c for c in ("date", "close") if c not in df.columns]
    if missing_required:
        errors.append("Missing required column(s): " + ", ".join(missing_required)
                      + ". Expected at least 'date' and 'close' (plus optional open, high, low, volume). "
                      + f"Found: {report['columns_raw']}")
        raise DatasetError("Missing required columns", report)

    # ---- timestamps -----------------------------------------------------
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        warnings.append(f"{bad_dates} row(s) had missing or unparseable timestamps and were dropped.")
    df["date"] = dates
    df = df[df["date"].notna()].copy()
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
        fixes.append("Removed timezone information from timestamps.")
    df["date"] = df["date"].dt.normalize()

    if not df["date"].is_monotonic_increasing:
        fixes.append("Rows were not in chronological order; sorted by date.")
    order = ["date"]
    if "series" in df.columns:  # NSE: prefer the regular equity series when a date has several
        df["_eq"] = (df["series"].astype(str).str.upper() == "EQ").astype(int)
        order.append("_eq")
    df = df.sort_values(order, kind="stable")
    dupes = int(df["date"].duplicated(keep="last").sum())
    if dupes:
        warnings.append(f"{dupes} duplicate timestamp(s) found; kept one row per date"
                        + (" (preferring series EQ)." if "series" in df.columns else " (the last one)."))
        df = df.drop_duplicates("date", keep="last")

    # ---- numeric values -------------------------------------------------
    present_price_cols = [c for c in PRICE_COLS if c in df.columns]
    num_cols = present_price_cols + [c for c in ("volume", "prev_close") if c in df.columns]
    for col in num_cols:
        before_na = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_numeric = int(df[col].isna().sum()) - before_na
        if non_numeric:
            warnings.append(f"{non_numeric} non-numeric value(s) in '{col}' were treated as missing.")

    for col in present_price_cols:
        nonpos = int((df[col] <= 0).sum())
        if nonpos:
            warnings.append(f"{nonpos} non-positive price(s) in '{col}' were treated as missing.")
            df.loc[df[col] <= 0, col] = np.nan
    if "volume" in df.columns:
        neg_vol = int((df["volume"] < 0).sum())
        if neg_vol:
            warnings.append(f"{neg_vol} negative volume value(s) were treated as missing.")
            df.loc[df["volume"] < 0, "volume"] = np.nan

    report["missing_by_column"] = {c: int(df[c].isna().sum()) for c in present_price_cols + (["volume"] if "volume" in df.columns else [])}

    close_na = int(df["close"].isna().sum())
    if close_na:
        df["close"] = df["close"].ffill(limit=5)
        still = int(df["close"].isna().sum())
        fixes.append(f"Forward-filled {close_na - still} missing close value(s) (max 5 consecutive). "
                     "Backfilling is never used because it would copy future prices into the past.")
        if still:
            warnings.append(f"Dropped {still} row(s) whose close could not be filled.")
        df = df[df["close"].notna()].copy()

    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
            fixes.append(f"Column '{col}' not provided; using close as a stand-in.")
        else:
            n = int(df[col].isna().sum())
            if n:
                df[col] = df[col].fillna(df["close"])
                fixes.append(f"Filled {n} missing '{col}' value(s) with the same day's close.")

    if "volume" not in df.columns or df["volume"].isna().all():
        df["volume"] = np.nan
        warnings.append("No volume data; volume-based features will be unavailable.")
    else:
        df["volume"] = df["volume"].ffill(limit=5)

    hl_bad = int((df["high"] < df["low"]).sum())
    if hl_bad:
        warnings.append(f"{hl_bad} row(s) have high < low; high/low were swapped for those rows.")
        swap = df["high"] < df["low"]
        df.loc[swap, ["high", "low"]] = df.loc[swap, ["low", "high"]].values

    df = df.reset_index(drop=True)
    df = _adjust_corporate_actions(df, fixes, warnings)

    jumps = int((df["close"].pct_change().abs() > 0.4).sum())
    if jumps:
        warnings.append(f"{jumps} single-day move(s) larger than 40% remain. They may be unadjusted corporate actions or bad ticks.")
    big_gaps = int((df["date"].diff().dt.days > 10).sum())
    if big_gaps:
        warnings.append(f"{big_gaps} gap(s) longer than 10 calendar days between consecutive rows.")

    df = df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    if len(df) < MIN_ROWS:
        errors.append(f"Only {len(df)} usable rows after cleaning. At least {MIN_ROWS} daily rows (about one year) "
                      "are needed to compute features and hold out a test period.")
        raise DatasetError("Not enough data", report)
    report["start"] = df["date"].iloc[0].strftime("%Y-%m-%d")
    report["end"] = df["date"].iloc[-1].strftime("%Y-%m-%d")
    report["status"] = "ok" if not warnings else "ok_with_warnings"
    return df, report


# --------------------------------------------------------------------------- reading stored data

def load_clean(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def summarize(df: pd.DataFrame) -> dict:
    rets = df["close"].pct_change().dropna()
    return {
        "rows": int(len(df)),
        "start": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "first_close": float(df["close"].iloc[0]),
        "last_close": float(df["close"].iloc[-1]),
        "min_close": float(df["close"].min()),
        "max_close": float(df["close"].max()),
        "mean_daily_return": float(rets.mean()),
        "daily_volatility": float(rets.std()),
        "annualized_volatility": float(rets.std() * np.sqrt(252)),
        "pct_up_days": float((rets > 0).mean()),
        "avg_volume": None if df["volume"].isna().all() else float(df["volume"].mean()),
        "has_volume": bool(df["volume"].notna().any()),
    }


def preview_series(df: pd.DataFrame, max_points: int = 800) -> dict:
    step = max(1, int(np.ceil(len(df) / max_points)))
    d = df.iloc[::step]
    return {
        "dates": d["date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": d["close"].round(4).tolist(),
        "volume": [None if pd.isna(v) else float(v) for v in d["volume"]],
    }
