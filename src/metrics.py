from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def log_returns(values: Iterable[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if len(values_arr) < 2:
        return np.zeros(0, dtype=np.float64)
    values_arr = np.clip(values_arr, 1e-12, None)
    return np.diff(np.log(values_arr))


def sharpe_ratio(returns: np.ndarray, steps_per_year: int) -> float:
    if returns.size == 0:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    if std == 0.0:
        return 0.0
    return mean / std * math.sqrt(steps_per_year)


def max_drawdown(values: Iterable[float]) -> float:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values_arr)
    drawdowns = (values_arr - running_max) / (running_max + 1e-12)
    return float(np.min(drawdowns))


def summarize(values: Iterable[float], steps_per_year: int) -> dict:
    values_arr = np.asarray(values, dtype=np.float64)
    returns = log_returns(values_arr)
    total_return = float(values_arr[-1] / values_arr[0] - 1.0) if values_arr.size >= 2 else 0.0
    return {
        "final_value": float(values_arr[-1]) if values_arr.size else 0.0,
        "total_return": total_return,
        "sharpe": sharpe_ratio(returns, steps_per_year),
        "max_drawdown": max_drawdown(values_arr),
        "volatility": float(np.std(returns)) if returns.size else 0.0,
    }


def buy_and_hold_series(prices: Iterable[float], initial_cash: float, fee: float, slippage: float) -> np.ndarray:
    price_arr = np.asarray(prices, dtype=np.float64)
    if price_arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    entry_price = price_arr[0] * (1.0 + fee + slippage)
    shares = initial_cash / entry_price
    return shares * price_arr
