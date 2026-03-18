from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.shared(obs)
        logits = self.policy_head(feats)
        value = self.value_head(feats).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        seed: int,
        device: torch.device,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device
        self.model = ActorCritic(obs_dim, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def action_logits(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, _ = self.model(obs_t)
        return logits.detach().cpu().numpy()

    def update(self, batch: RolloutBatch, epochs: int, minibatch_size: int) -> None:
        n = batch.observations.shape[0]
        if minibatch_size > n:
            minibatch_size = n
        indices = np.arange(n)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, n, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                obs = batch.observations[mb_idx]
                actions = batch.actions[mb_idx]
                old_logprobs = batch.logprobs[mb_idx]
                returns = batch.returns[mb_idx]
                advantages = batch.advantages[mb_idx]

                logits, values = self.model(obs)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - values).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0

    for idx in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[idx])
        delta = rewards[idx] + gamma * next_value * nonterminal - values[idx]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[idx] = last_gae
        next_value = values[idx]

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def make_batch(
    observations: list[np.ndarray],
    actions: list[int],
    logprobs: list[float],
    returns: np.ndarray,
    advantages: np.ndarray,
    device: torch.device,
) -> RolloutBatch:
    obs_t = torch.as_tensor(np.array(observations), dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    logprobs_t = torch.as_tensor(logprobs, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return RolloutBatch(obs_t, actions_t, logprobs_t, returns_t, adv_t)


def clone_with_noise(agent: PPOAgent, noise_scale: float, seed: int) -> PPOAgent:
    device = agent.device
    clone = PPOAgent(
        obs_dim=agent.model.shared[0].in_features,
        n_actions=agent.model.policy_head.out_features,
        lr=agent.optimizer.param_groups[0]["lr"],
        clip_range=agent.clip_range,
        ent_coef=agent.ent_coef,
        vf_coef=agent.vf_coef,
        max_grad_norm=agent.max_grad_norm,
        seed=seed,
        device=device,
    )
    clone.model.load_state_dict(agent.model.state_dict())
    with torch.no_grad():
        for param in clone.model.parameters():
            param.add_(noise_scale * torch.randn_like(param))
    return clone
