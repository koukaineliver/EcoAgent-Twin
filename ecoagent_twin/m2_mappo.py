"""Train a shared masked actor and centralized critic for event-driven routing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from statistics import fmean
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence


class Decision(Protocol):
    actions: tuple
    actor_features: tuple[float, ...]
    critic_features: tuple[float, ...]
    action_features: tuple[tuple[float, ...], ...]


class Environment(Protocol):
    def observation(self) -> Decision | None: ...

    def step(self, action_index: int) -> float: ...

    def result(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrainingConfig:
    hidden_size: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    max_gradient_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 128
    validation_interval: int = 5


@dataclass(frozen=True)
class Transition:
    decision: Decision
    action: int
    log_probability: float
    value: float
    reward: float


class MAPPO(nn.Module):
    """Score structurally feasible actions with one actor shared by every vehicle."""

    def __init__(self, actor_dim: int, critic_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.dimensions = (actor_dim, critic_dim, action_dim, hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(actor_dim + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def distribution(self, local: Tensor, candidates: Tensor, mask: Tensor) -> Categorical:
        local_per_action = local.unsqueeze(1).expand(-1, candidates.size(1), -1)
        joined = torch.cat((local_per_action, candidates), dim=-1)
        logits = self.actor(joined).squeeze(-1).masked_fill(~mask, -torch.inf)
        return Categorical(logits=logits)

    def _decision_distribution(self, decision: Decision) -> Categorical:
        count = len(decision.actions)
        if count == 0 or count != len(decision.action_features):
            raise ValueError("A decision needs a feature row for each feasible action")
        local = torch.tensor([decision.actor_features], dtype=torch.float32)
        candidates = torch.tensor([decision.action_features], dtype=torch.float32)
        mask = torch.ones((1, count), dtype=torch.bool)
        return self.distribution(local, candidates, mask)

    @torch.inference_mode()
    def choose(self, decision: Decision) -> int:
        return int(self._decision_distribution(decision).logits.argmax(dim=-1).item())

    @torch.inference_mode()
    def sample(self, decision: Decision) -> tuple[int, float, float]:
        distribution = self._decision_distribution(decision)
        action = distribution.sample()
        critic_input = torch.tensor([decision.critic_features], dtype=torch.float32)
        value = self.critic(critic_input).item()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value)

    def save(self, path: str | Path) -> None:
        payload = {
            "schema": "ecoagent-twin-m2-mappo-v1",
            "dimensions": self.dimensions,
            "state_dict": self.state_dict(),
        }
        with Path(path).open("xb") as destination:
            torch.save(payload, destination)

    @classmethod
    def load(cls, path: str | Path) -> MAPPO:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema"] != "ecoagent-twin-m2-mappo-v1":
            raise ValueError("Unsupported M2 MAPPO checkpoint schema")
        policy = cls(*payload["dimensions"])
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy


def episode(
    policy: MAPPO, env: Environment, *, stochastic: bool
) -> tuple[list[Transition], float, float]:
    trajectory = []
    total_reward = 0.0
    while (decision := env.observation()) is not None:
        if stochastic:
            action, log_probability, value = policy.sample(decision)
            reward = float(env.step(action))
            trajectory.append(Transition(decision, action, log_probability, value, reward))
        else:
            reward = float(env.step(policy.choose(decision)))
        total_reward += reward
    return trajectory, float(env.result()["metrics"]["service_rate"]), total_reward


def gae(trajectory: list[Transition], config: TrainingConfig) -> list[float]:
    advantages = []
    next_value = 0.0
    next_advantage = 0.0
    for step in reversed(trajectory):
        delta = step.reward + config.gamma * next_value - step.value
        next_advantage = delta + config.gamma * config.gae_lambda * next_advantage
        advantages.append(next_advantage)
        next_value = step.value
    return advantages[::-1]


def optimize(
    policy: MAPPO,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    episode_advantages: list[float],
    config: TrainingConfig,
) -> float:
    local = torch.tensor([step.decision.actor_features for step in transitions], dtype=torch.float32)
    global_state = torch.tensor(
        [step.decision.critic_features for step in transitions], dtype=torch.float32
    )
    candidate_rows = [
        torch.tensor(step.decision.action_features, dtype=torch.float32) for step in transitions
    ]
    candidates = pad_sequence(candidate_rows, batch_first=True)
    lengths = torch.tensor([row.size(0) for row in candidate_rows])
    mask = torch.arange(candidates.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    actions = torch.tensor([step.action for step in transitions], dtype=torch.long)
    old_log_probs = torch.tensor([step.log_probability for step in transitions], dtype=torch.float32)
    advantages = torch.tensor(episode_advantages, dtype=torch.float32)
    returns = advantages + torch.tensor([step.value for step in transitions], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    losses = []
    for _ in range(config.epochs):
        permutation = torch.randperm(len(transitions))
        for indexes in permutation.split(config.minibatch_size):
            distribution = policy.distribution(local[indexes], candidates[indexes], mask[indexes])
            ratio = (distribution.log_prob(actions[indexes]) - old_log_probs[indexes]).exp()
            clipped_ratio = ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
            batch_advantages = advantages[indexes]
            policy_loss = -torch.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            values = policy.critic(global_state[indexes]).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns[indexes])
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
            optimizer.step()
            losses.append(float(loss.detach().item()))
    return fmean(losses)


def validation(
    policy: MAPPO, factories: list[Callable[[], Environment]]
) -> tuple[float, float]:
    policy.eval()
    outcomes = [episode(policy, factory(), stochastic=False)[1:] for factory in factories]
    policy.train()
    return fmean(service for service, _ in outcomes), fmean(reward for _, reward in outcomes)


def train(
    probe_env: Environment,
    training_envs: list[Callable[[], Environment]],
    validation_envs: list[Callable[[], Environment]],
    seed: int,
    updates: int = 30,
    episodes_per_update: int = 16,
) -> tuple[MAPPO, dict[str, Any]]:
    if min(updates, episodes_per_update) < 1 or not training_envs or not validation_envs:
        raise ValueError("Training needs positive run counts and nonempty train and validation sets")
    probe = probe_env.observation()
    if probe is None or not probe.action_features:
        raise ValueError("The probe must expose a decision with candidate action features")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    config = TrainingConfig()
    policy = MAPPO(
        len(probe.actor_features),
        len(probe.critic_features),
        len(probe.action_features[0]),
        config.hidden_size,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    case_order: list[int] = []
    best_weights = None
    best_validation = (-torch.inf, -torch.inf)
    best_update = 0
    history = []

    for update in range(1, updates + 1):
        transitions = []
        advantages = []
        services = []
        rewards = []
        for _ in range(episodes_per_update):
            if not case_order:
                case_order = list(range(len(training_envs)))
                rng.shuffle(case_order)
            env = training_envs[case_order.pop()]()
            trajectory, service, reward = episode(policy, env, stochastic=True)
            transitions.extend(trajectory)
            advantages.extend(gae(trajectory, config))
            services.append(service)
            rewards.append(reward)
        if not transitions:
            raise ValueError(f"Training update {update} received no routing decisions")
        update_receipt = {
            "update": update,
            "training_decisions": len(transitions),
            "training_mean_service": fmean(services),
            "training_mean_episode_return": fmean(rewards),
            "mean_ppo_loss": optimize(policy, optimizer, transitions, advantages, config),
        }
        if update in (1, updates) or update % config.validation_interval == 0:
            score = validation(policy, validation_envs)
            update_receipt.update(
                validation_mean_service=score[0], validation_mean_episode_return=score[1]
            )
            if score > best_validation:
                best_validation = score
                best_update = update
                best_weights = {
                    name: weight.detach().clone() for name, weight in policy.state_dict().items()
                }
        history.append(update_receipt)

    if best_weights is None:
        raise ValueError("No usable validation checkpoint was obtained")
    policy.load_state_dict(best_weights)
    policy.eval()
    return policy, {
        "algorithm": "parameter-shared masked PPO with a centralized value function",
        "seed": seed,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "deterministic_torch_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config": asdict(config),
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "training_cases": len(training_envs),
        "validation_cases": len(validation_envs),
        "discount_unit": "routing decision",
        "checkpoint_selection": "highest validation mean service, then mean return, then earliest update",
        "selected_update": best_update,
        "selected_validation_mean_service": best_validation[0],
        "selected_validation_mean_episode_return": best_validation[1],
        "history": history,
    }
