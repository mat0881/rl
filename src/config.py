from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path("data/btc_1m_parquet")
    symbol: str = "BTCUSDT"
    start: str = "2025-11-16"
    end: str = "2026-02-16"


@dataclass
class EnvConfig:
    window_size: int = 60
    initial_cash: float = 100000.0
    position_size: float = 0.01
    max_shares: int = 10
    fee: float = 0.0004
    slippage: float = 0.0001
    episode_length: int = 1440
    reward_mode: str = "pnl"
    reward_scale: float = 1.0
    seed: int = 42
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    cci_period: int = 20


@dataclass
class PodracerConfig:
    population_size: int = 4
    elite_size: int = 2
    generations: int = 5
    train_ratio: float = 0.8
    rollout_steps: int = 2048
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    mutation_scale: float = 0.02
    seed: int = 42
    eval_episodes: int = 3
    run_dir: Path = Path("runs")


@dataclass
class TrainConfig:
    epochs: int = 10
    gamma: float = 0.99
    lr: float = 1e-3
    seed: int = 42
