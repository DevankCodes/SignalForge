"""Download real daily OHLCV data from Yahoo Finance into CSVs SignalForge can upload.

Usage (run on your own laptop, needs internet):
    pip install yfinance
    python scripts/fetch_yfinance.py RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ^NSEI
Files are written to real_data/<SYMBOL>.csv
"""
import sys
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).resolve().parent.parent / "real_data"


def main(symbols, start="2012-01-01"):
    OUT.mkdir(exist_ok=True)
    for sym in symbols:
        df = yf.download(sym, start=start, auto_adjust=True, progress=False)
        if df.empty:
            print(f"{sym}: no data returned, skipped")
            continue
        if hasattr(df.columns, "levels"):  # newer yfinance returns a column MultiIndex
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        path = OUT / f"{sym.replace('^', 'IDX_')}.csv"
        df.to_csv(path, index=False)
        print(f"{sym}: {len(df)} rows -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"])
