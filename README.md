# FinRL-Podracer BTC 1m (Repro)

This folder contains a minimal data pipeline to download BTC 1-minute OHLCV data from Binance and store it as daily Parquet files. The rest of the reproduction will build on top of these files.

## Setup

```bash
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
python -m pip install -r requirements.txt
```

## Download BTC 1m data

Example for the last 3 months (UTC):

```bash
python data\download_binance_klines.py --symbol BTCUSDT --interval 1m --start 2025-11-16 --end 2026-02-16 --out_dir data\btc_1m_parquet
```

## Minimal training run

```bash
python src\train.py
```

## Podracer-style training (PPO + generational evolution + ensemble)

```bash
python src\podracer_train.py
```

Outputs (metrics and checkpoints) are stored under `runs/run_YYYYMMDD_HHMMSS/`.

### State features (paper-aligned)

The environment uses the paper's state definition for a single asset:
- account balance
- shares held
- close price
- MACD, RSI, CCI

### Output schema

Each Parquet file contains these columns:
- timestamp_ms (int)
- datetime (datetime64[ns, UTC])
- open, high, low, close (float)
- volume (float)
- close_time_ms (int)
- quote_volume (float)
- trades (int)
- taker_buy_base (float)
- taker_buy_quote (float)
- symbol (str)

## Notes

- Data is pulled from the public Binance REST API (no key required).
- Files are saved as one Parquet per day, named YYYY-MM-DD.parquet.
- Fees/slippage target: fees 0.04%, slippage 0.01% (will be used in the training pipeline).
