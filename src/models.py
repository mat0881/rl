from __future__ import annotations

import numpy as np


class LinearPolicy:
    def __init__(self, obs_dim: int, n_actions: int, rng: np.random.Generator) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rng = rng
        self.weights = rng.normal(scale=0.01, size=(obs_dim, n_actions))
        self.bias = np.zeros(n_actions, dtype=np.float64)

    def action_distribution(self, obs: np.ndarray) -> np.ndarray:
        logits = obs @ self.weights + self.bias
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def sample_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        probs = self.action_distribution(obs)
        action = int(self.rng.choice(self.n_actions, p=probs))
        return action, probs

    def grad_logp(self, obs: np.ndarray, action: int, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad_logits = -probs
        grad_logits[action] += 1.0
        grad_w = np.outer(obs, grad_logits)
        grad_b = grad_logits
        return grad_w, grad_b
