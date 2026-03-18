from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class TradingEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 60,
        initial_cash: float = 100000.0,
        position_size: float = 1.0,
        max_shares: int = 10,
        fee: float = 0.0004,
        slippage: float = 0.0001,
        episode_length: int | None = None,
        random_start: bool = True,
        reward_mode: str = "log",
        reward_scale: float = 1.0,
        seed: int = 42,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        cci_period: int = 20,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.max_shares = max_shares
        self.fee = fee
        self.slippage = slippage
        self.episode_length = episode_length
        self.random_start = random_start
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.cci_period = cci_period

        self.prices = self.data["close"].astype("float64").to_numpy()
        self.features = self._build_features(self.data)
        self.n_actions = self.max_shares * 2 + 1
        self.action_offset = self.max_shares

        self._idx = 0
        self._episode_end = len(self.prices) - 1
        self._start_idx = 0
        self._cash = initial_cash
        self._holdings = 0.0

    @property
    def observation_size(self) -> int:
        return self.features.shape[1] + 2

    def reset(self) -> np.ndarray:
        if self.episode_length is None:
            start_idx = self.window_size - 1
            self._episode_end = len(self.prices) - 1
        else:
            min_start = self.window_size - 1
            max_start = len(self.prices) - self.episode_length - 1
            if max_start <= min_start:
                raise ValueError("episode_length is too large for the dataset")
            if self.random_start:
                start_idx = int(self.rng.integers(min_start, max_start + 1))
            else:
                start_idx = min_start
            self._episode_end = start_idx + self.episode_length - 1

        self._idx = start_idx
        self._start_idx = start_idx
        self._cash = self.initial_cash
        self._holdings = 0.0
        return self._get_observation()

    def step(self, action: int) -> StepResult:
        if action < -self.max_shares or action > self.max_shares:
            raise ValueError("action must be within [-max_shares, max_shares]")

        price = float(self.prices[self._idx])
        prev_value = self._portfolio_value(price)
        self._apply_action(action, price)

        self._idx += 1
        done = self._idx >= self._episode_end
        next_price = float(self.prices[self._idx])
        next_value = self._portfolio_value(next_price)
        reward = self._compute_reward(prev_value, next_value)

        obs = self._get_observation()
        info = {
            "portfolio_value": next_value,
            "cash": self._cash,
            "holdings": self._holdings,
            "shares": self._holdings,
            "price": next_price,
            "index": self._idx,
        }
        return StepResult(obs, reward, done, info)

    def current_price(self) -> float:
        return float(self.prices[self._idx])

    def _apply_action(self, trade_shares: int, price: float) -> None:
        if trade_shares == 0:
            return

        trade_units = trade_shares * self.position_size

        if trade_units > 0:
            cost = trade_units * price * (1.0 + self.fee + self.slippage)
            if self._cash < cost:
                return
            self._cash -= cost
            self._holdings += trade_units
        else:
            proceeds = -trade_units * price * (1.0 - self.fee - self.slippage)
            if self._holdings < -trade_units:
                return
            self._cash += proceeds
            self._holdings += trade_units

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._holdings * price

    def _get_observation(self) -> np.ndarray:
        price = float(self.prices[self._idx])
        holdings_value = self._holdings * price
        extra = np.array(
            [
                float(self._cash / self.initial_cash),
                float(holdings_value / self.initial_cash),
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.features[self._idx], extra])

    def _compute_reward(self, prev_value: float, next_value: float) -> float:
        if self.reward_mode == "pnl":
            reward = next_value - prev_value
        elif self.reward_mode == "log":
            if prev_value <= 0.0 or next_value <= 0.0:
                reward = -1.0
            else:
                reward = float(np.log(next_value / prev_value))
        else:
            raise ValueError("reward_mode must be 'pnl' or 'log'")
        return reward * self.reward_scale

    def _build_features(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].astype("float64")
        high = data["high"].astype("float64")
        low = data["low"].astype("float64")

        close_norm = close / close.iloc[0] - 1.0

        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.macd_signal, adjust=False).mean()

        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0

        tp = (high + low + close) / 3.0
        tp_sma = tp.rolling(window=self.cci_period).mean()
        mean_dev = (tp - tp_sma).abs().rolling(window=self.cci_period).mean()
        cci = ((tp - tp_sma) / (0.015 * (mean_dev + 1e-8))).fillna(0.0) / 100.0

        raw = pd.DataFrame(
            {
                "close": close_norm,
                "macd": macd,
                "macd_signal": macd_signal,
                "rsi": rsi,
                "cci": cci,
            }
        )

        return raw.fillna(0.0).to_numpy(dtype=np.float32)
