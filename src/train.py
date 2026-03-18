from __future__ import annotations

import numpy as np

from config import DataConfig, EnvConfig, TrainConfig
from data import load_parquet_dir
from env import TradingEnv
from models import LinearPolicy


def compute_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def run_episode(env: TradingEnv, policy: LinearPolicy) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[float], float]:
    obs = env.reset()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    probs_list: list[np.ndarray] = []
    rewards: list[float] = []
    done = False
    last_value = 0.0

    while not done:
        action_idx, probs = policy.sample_action(obs)
        action = action_idx - env.action_offset
        result = env.step(action)

        observations.append(obs)
        actions.append(action_idx)
        probs_list.append(probs)
        rewards.append(result.reward)
        last_value = result.info["portfolio_value"]

        obs = result.observation
        done = result.done

    return observations, actions, probs_list, rewards, last_value


def train() -> None:
    data_cfg = DataConfig()
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()

    data = load_parquet_dir(data_cfg.data_dir, data_cfg.symbol, data_cfg.start, data_cfg.end)
    if len(data) <= env_cfg.window_size:
        raise ValueError("Not enough data for the requested window size.")

    env = TradingEnv(
        data,
        window_size=env_cfg.window_size,
        initial_cash=env_cfg.initial_cash,
        position_size=env_cfg.position_size,
        max_shares=env_cfg.max_shares,
        fee=env_cfg.fee,
        slippage=env_cfg.slippage,
        reward_mode=env_cfg.reward_mode,
        reward_scale=env_cfg.reward_scale,
        seed=env_cfg.seed,
        macd_fast=env_cfg.macd_fast,
        macd_slow=env_cfg.macd_slow,
        macd_signal=env_cfg.macd_signal,
        rsi_period=env_cfg.rsi_period,
        cci_period=env_cfg.cci_period,
    )

    rng = np.random.default_rng(train_cfg.seed)
    policy = LinearPolicy(env.observation_size, env.n_actions, rng)

    for epoch in range(train_cfg.epochs):
        observations, actions, probs_list, rewards, final_value = run_episode(env, policy)
        returns = compute_returns(rewards, train_cfg.gamma)

        for obs, action, probs, ret in zip(observations, actions, probs_list, returns):
            grad_w, grad_b = policy.grad_logp(obs, action, probs)
            policy.weights += train_cfg.lr * ret * grad_w
            policy.bias += train_cfg.lr * ret * grad_b

        print(f"Epoch {epoch + 1:02d} | Final portfolio value: {final_value:,.2f}")


if __name__ == "__main__":
    train()
