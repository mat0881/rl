from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DataConfig, EnvConfig, PodracerConfig
from data import load_parquet_dir, split_data_by_time
from env import TradingEnv
from logger import CSVLogger
from metrics import buy_and_hold_series, summarize
from ppo import PPOAgent, compute_gae, make_batch, clone_with_noise

STEPS_PER_YEAR = 365 * 24 * 60


def collect_rollout(
    env: TradingEnv, agent: PPOAgent, rollout_steps: int
) -> tuple[list[np.ndarray], list[int], list[float], list[float], list[float], list[bool]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    logprobs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []

    obs = env.reset()
    while len(rewards) < rollout_steps:
        action_idx, logprob, value = agent.act(obs)
        result = env.step(action_idx - env.action_offset)

        observations.append(obs)
        actions.append(action_idx)
        logprobs.append(logprob)
        values.append(value)
        rewards.append(result.reward)
        dones.append(result.done)

        obs = result.observation
        if result.done:
            obs = env.reset()

    return observations, actions, logprobs, values, rewards, dones


def run_episode(env: TradingEnv, policy_fn) -> tuple[np.ndarray, np.ndarray]:
    obs = env.reset()
    values = [env.initial_cash]
    prices = [env.current_price()]
    done = False

    while not done:
        action_idx = int(policy_fn(obs))
        result = env.step(action_idx - env.action_offset)
        values.append(result.info["portfolio_value"])
        prices.append(result.info["price"])
        obs = result.observation
        done = result.done

    return np.asarray(values, dtype=np.float64), np.asarray(prices, dtype=np.float64)


def evaluate_policy(env_cfg: EnvConfig, data: pd.DataFrame, policy_fn, episodes: int) -> dict:
    env = TradingEnv(
        data,
        window_size=env_cfg.window_size,
        initial_cash=env_cfg.initial_cash,
        position_size=env_cfg.position_size,
        max_shares=env_cfg.max_shares,
        fee=env_cfg.fee,
        slippage=env_cfg.slippage,
        episode_length=env_cfg.episode_length,
        random_start=True,
        reward_mode=env_cfg.reward_mode,
        reward_scale=env_cfg.reward_scale,
        seed=env_cfg.seed,
        macd_fast=env_cfg.macd_fast,
        macd_slow=env_cfg.macd_slow,
        macd_signal=env_cfg.macd_signal,
        rsi_period=env_cfg.rsi_period,
        cci_period=env_cfg.cci_period,
    )

    metrics = []
    for _ in range(episodes):
        values, _ = run_episode(env, policy_fn)
        metrics.append(summarize(values, STEPS_PER_YEAR))

    return _average_metrics(metrics)


def evaluate_agent(env_cfg: EnvConfig, data: pd.DataFrame, agent: PPOAgent, episodes: int) -> dict:
    def policy_fn(obs: np.ndarray) -> int:
        logits = agent.action_logits(obs)
        return int(np.argmax(logits))

    return evaluate_policy(env_cfg, data, policy_fn, episodes)


def evaluate_ensemble(env_cfg: EnvConfig, data: pd.DataFrame, agents: list[PPOAgent], episodes: int) -> dict:
    def policy_fn(obs: np.ndarray) -> int:
        logits = np.mean([agent.action_logits(obs) for agent in agents], axis=0)
        return int(np.argmax(logits))

    return evaluate_policy(env_cfg, data, policy_fn, episodes)


def evaluate_buy_and_hold(env_cfg: EnvConfig, data: pd.DataFrame, episodes: int) -> dict:
    env = TradingEnv(
        data,
        window_size=env_cfg.window_size,
        initial_cash=env_cfg.initial_cash,
        position_size=env_cfg.position_size,
        max_shares=env_cfg.max_shares,
        fee=env_cfg.fee,
        slippage=env_cfg.slippage,
        episode_length=env_cfg.episode_length,
        random_start=True,
        reward_mode=env_cfg.reward_mode,
        reward_scale=env_cfg.reward_scale,
        seed=env_cfg.seed,
        macd_fast=env_cfg.macd_fast,
        macd_slow=env_cfg.macd_slow,
        macd_signal=env_cfg.macd_signal,
        rsi_period=env_cfg.rsi_period,
        cci_period=env_cfg.cci_period,
    )

    metrics = []
    for _ in range(episodes):
        _, prices = run_episode(env, lambda obs: env.action_offset)
        values = buy_and_hold_series(prices, env.initial_cash, env.fee, env.slippage)
        metrics.append(summarize(values, STEPS_PER_YEAR))

    return _average_metrics(metrics)


def _average_metrics(metrics: list[dict]) -> dict:
    if not metrics:
        return {"final_value": 0.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "volatility": 0.0}
    avg = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
    return avg


def _prepare_run_dir(base_dir: Path) -> Path:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_elites(run_dir: Path, elites: list[PPOAgent], generation: int) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for idx, agent in enumerate(elites):
        torch.save(agent.model.state_dict(), ckpt_dir / f"gen_{generation:02d}_elite_{idx}.pt")


def train() -> None:
    data_cfg = DataConfig()
    env_cfg = EnvConfig()
    pod_cfg = PodracerConfig()

    data = load_parquet_dir(data_cfg.data_dir, data_cfg.symbol, data_cfg.start, data_cfg.end)
    train_data, val_data = split_data_by_time(data, pod_cfg.train_ratio)

    device = torch.device("cpu")
    env = TradingEnv(
        train_data,
        window_size=env_cfg.window_size,
        initial_cash=env_cfg.initial_cash,
        position_size=env_cfg.position_size,
        max_shares=env_cfg.max_shares,
        fee=env_cfg.fee,
        slippage=env_cfg.slippage,
        episode_length=env_cfg.episode_length,
        random_start=True,
        reward_mode=env_cfg.reward_mode,
        reward_scale=env_cfg.reward_scale,
        seed=env_cfg.seed,
        macd_fast=env_cfg.macd_fast,
        macd_slow=env_cfg.macd_slow,
        macd_signal=env_cfg.macd_signal,
        rsi_period=env_cfg.rsi_period,
        cci_period=env_cfg.cci_period,
    )

    run_dir = _prepare_run_dir(pod_cfg.run_dir)
    logger = CSVLogger(
        run_dir / "metrics.csv",
        [
            "generation",
            "best_final_value",
            "best_total_return",
            "best_sharpe",
            "best_max_drawdown",
            "best_volatility",
            "ensemble_final_value",
            "ensemble_total_return",
            "ensemble_sharpe",
            "ensemble_max_drawdown",
            "ensemble_volatility",
            "buyhold_final_value",
            "buyhold_total_return",
            "buyhold_sharpe",
            "buyhold_max_drawdown",
            "buyhold_volatility",
        ],
    )

    population: list[PPOAgent] = []
    for idx in range(pod_cfg.population_size):
        agent = PPOAgent(
            obs_dim=env.observation_size,
            n_actions=env.n_actions,
            lr=pod_cfg.lr,
            clip_range=pod_cfg.clip_range,
            ent_coef=pod_cfg.ent_coef,
            vf_coef=pod_cfg.vf_coef,
            max_grad_norm=pod_cfg.max_grad_norm,
            seed=pod_cfg.seed + idx,
            device=device,
        )
        population.append(agent)

    for generation in range(pod_cfg.generations):
        scores = []
        metrics_cache = []
        for agent in population:
            observations, actions, logprobs, values, rewards, dones = collect_rollout(
                env, agent, pod_cfg.rollout_steps
            )
            advantages, returns = compute_gae(rewards, values, dones, pod_cfg.gamma, pod_cfg.gae_lambda)
            batch = make_batch(observations, actions, logprobs, returns, advantages, device)
            agent.update(batch, pod_cfg.ppo_epochs, pod_cfg.minibatch_size)

            metrics = evaluate_agent(env_cfg, val_data, agent, pod_cfg.eval_episodes)
            metrics_cache.append(metrics)
            scores.append(metrics["final_value"])

        ranked = sorted(zip(population, metrics_cache, scores), key=lambda item: item[2], reverse=True)
        elites = [agent for agent, _, _ in ranked[: pod_cfg.elite_size]]
        best_metrics = ranked[0][1]

        ensemble_metrics = evaluate_ensemble(env_cfg, val_data, elites, pod_cfg.eval_episodes)
        buyhold_metrics = evaluate_buy_and_hold(env_cfg, val_data, pod_cfg.eval_episodes)

        logger.log(
            {
                "generation": generation + 1,
                "best_final_value": best_metrics["final_value"],
                "best_total_return": best_metrics["total_return"],
                "best_sharpe": best_metrics["sharpe"],
                "best_max_drawdown": best_metrics["max_drawdown"],
                "best_volatility": best_metrics["volatility"],
                "ensemble_final_value": ensemble_metrics["final_value"],
                "ensemble_total_return": ensemble_metrics["total_return"],
                "ensemble_sharpe": ensemble_metrics["sharpe"],
                "ensemble_max_drawdown": ensemble_metrics["max_drawdown"],
                "ensemble_volatility": ensemble_metrics["volatility"],
                "buyhold_final_value": buyhold_metrics["final_value"],
                "buyhold_total_return": buyhold_metrics["total_return"],
                "buyhold_sharpe": buyhold_metrics["sharpe"],
                "buyhold_max_drawdown": buyhold_metrics["max_drawdown"],
                "buyhold_volatility": buyhold_metrics["volatility"],
            }
        )

        print(
            "Generation %02d | Best value: %.2f | Ensemble: %.2f | Buy&Hold: %.2f"
            % (
                generation + 1,
                best_metrics["final_value"],
                ensemble_metrics["final_value"],
                buyhold_metrics["final_value"],
            )
        )

        _save_elites(run_dir, elites, generation + 1)

        next_population = elites.copy()
        while len(next_population) < pod_cfg.population_size:
            parent = elites[len(next_population) % len(elites)]
            child = clone_with_noise(parent, pod_cfg.mutation_scale, pod_cfg.seed + generation + len(next_population))
            next_population.append(child)
        population = next_population


if __name__ == "__main__":
    train()
