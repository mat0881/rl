import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000


def parse_date(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def to_ms(value: dt.datetime) -> int:
    return int(value.timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_LIMIT,
    }
    for attempt in range(5):
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (418, 429):
            time.sleep(2 + attempt)
            continue
        response.raise_for_status()
    response.raise_for_status()


def klines_to_frame(symbol: str, raw: list[list]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(raw, columns=columns)
    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    frame[float_cols] = frame[float_cols].astype("float64")
    frame["trades"] = frame["trades"].astype("int64")
    frame["datetime"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.rename(columns={"open_time": "timestamp_ms", "close_time": "close_time_ms"})
    frame["symbol"] = symbol
    frame = frame.drop(columns=["ignore"])
    return frame


def download_range(symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    all_rows: list[pd.DataFrame] = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        raw = fetch_klines(symbol, interval, cursor_ms, end_ms)
        if not raw:
            break
        batch = klines_to_frame(symbol, raw)
        all_rows.append(batch)
        last_close = int(raw[-1][6])
        cursor_ms = last_close + 1
        if cursor_ms <= batch["timestamp_ms"].iloc[-1]:
            cursor_ms = int(batch["timestamp_ms"].iloc[-1]) + 60000
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def write_daily_parquet(frame: pd.DataFrame, out_dir: Path) -> None:
    frame = frame.sort_values("timestamp_ms")
    frame["date"] = frame["datetime"].dt.strftime("%Y-%m-%d")
    for day, day_frame in frame.groupby("date"):
        output_path = out_dir / f"{day}.parquet"
        day_frame = day_frame.drop(columns=["date"])
        day_frame.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance klines and store as daily parquet files.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD in UTC")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD in UTC (exclusive)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    if end <= start:
        raise ValueError("end must be after start")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = download_range(args.symbol, args.interval, start, end)
    if frame.empty:
        print("No data returned from Binance.")
        return

    write_daily_parquet(frame, out_dir)
    print(f"Saved {len(frame)} rows to {out_dir}")


if __name__ == "__main__":
    main()
