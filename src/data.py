from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp_ms",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "symbol",
}


def _iter_parquet_files(data_dir: Path) -> Iterable[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    return sorted(path for path in data_dir.glob("*.parquet") if path.is_file())


def load_parquet_dir(data_dir: Path, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    files = list(_iter_parquet_files(data_dir))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
    data = data.sort_values("timestamp_ms")
    data = data[data["symbol"] == symbol]

    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        data = data[data["datetime"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        data = data[data["datetime"] < end_ts]

    return data.reset_index(drop=True)


def split_data_by_time(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    cutoff = int(len(data) * train_ratio)
    train = data.iloc[:cutoff].reset_index(drop=True)
    test = data.iloc[cutoff:].reset_index(drop=True)
    return train, test
