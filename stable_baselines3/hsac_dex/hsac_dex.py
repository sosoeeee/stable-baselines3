from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch as th
from torch.nn import functional as F

from stable_baselines3.hsac import HSAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.hsac_dex.demo_buffer import DemoBuffer


class HSAC_DEX(HSAC):
    """DEX-style hybrid actor-critic built on HSAC."""

    def __init__(
        self,
        *args,
        demo_path: Optional[str] = None,
        demo_batch_size: int = 256,
        demo_aux_weight: float = 1.0,
        demo_k: int = 5,
        demo_id_margin: float = 1.0,
        **kwargs,
    ):
        self.demo_path = demo_path
        self.demo_batch_size = demo_batch_size
        self.demo_aux_weight = demo_aux_weight
        self.demo_k = demo_k
        self.demo_id_margin = demo_id_margin
        self._demo_buffer: Optional[DemoBuffer] = None
        self.actor_target = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.demo_path is not None:
            self._demo_buffer = DemoBuffer.from_npz(self.demo_path, self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.set_training_mode(False)

    def _deterministic_action(self, actor, obs: th.Tensor) -> dict:
        logits = actor.get_task_dist_params(obs)
        discrete_action = th.argmax(logits, dim=1)
        mean_actions, log_std, kwargs = actor.get_param_dist_params(obs, discrete_action)
        continuous_action = actor.param_action_dist.actions_from_params(
            mean_actions, log_std, deterministic=True, **kwargs
        )
        return {self.d_key: discrete_action, self.c_key: continuous_action}

    def _compute_propagated_actions(
        self,
        obs: th.Tensor,
        obs_demo: th.Tensor,
        demo_ids: th.Tensor,
        demo_params: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        k = min(self.demo_k, obs_demo.shape[0])
        l2_pair = th.cdist(obs, obs_demo)
        topk_values, topk_indices = l2_pair.topk(k, dim=1, largest=False)
        topk_weights = F.softmax(-topk_values, dim=1)

        topk_ids = demo_ids[topk_indices]  # (batch, k)
        topk_params = demo_params[topk_indices]  # (batch, k, param_dim)

        batch_size = obs.shape[0]
        n_actions = self.actor.n_discrete_actions
        id_weight = th.zeros((batch_size, n_actions), device=obs.device)
        for i in range(k):
            id_weight.scatter_add_(1, topk_ids[:, i].unsqueeze(1), topk_weights[:, i].unsqueeze(1))
        prop_ids = th.argmax(id_weight, dim=1)

        prop_params = th.zeros((batch_size, demo_params.shape[1]), device=obs.device)
        for i in range(batch_size):
            mask = topk_ids[i] == prop_ids[i]
            if mask.any():
                weights = topk_weights[i][mask]
                values = topk_params[i][mask]
            else:
                weights = topk_weights[i]
                values = topk_params[i]
            weights = weights / (weights.sum() + 1e-8)
            prop_params[i] = (weights.unsqueeze(1) * values).sum(dim=0)

        return prop_ids, prop_params

    def _hybrid_act_dist(
        self,
        pred_ids: th.Tensor,
        pred_params: th.Tensor,
        target_ids: th.Tensor,
        target_params: th.Tensor,
    ) -> th.Tensor:
        param_dim = max(1, target_params.shape[1])
        param_dist = -((pred_params - target_params).pow(2).sum(dim=1) / param_dim)
        id_mismatch = (pred_ids != target_ids).float() * (-self.demo_id_margin)
        return param_dist + id_mismatch

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        last_actor_loss = 0.0
        last_critic_loss = 0.0

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            if self._demo_buffer is None:
                raise RuntimeError("demo_path must be provided for HSAC_DEX")
            demo_obs, demo_actions = self._demo_buffer.sample(self.demo_batch_size)
            demo_obs_t = th.as_tensor(demo_obs, device=self.device)
            demo_ids = th.as_tensor(demo_actions["id"], device=self.device).long().flatten()
            demo_params = th.as_tensor(demo_actions["params"], device=self.device)

            with th.no_grad():
                next_actions = self._deterministic_action(self.actor_target, replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                prop_ids, prop_params = self._compute_propagated_actions(
                    replay_data.next_observations, demo_obs_t, demo_ids, demo_params
                )
                act_dist = self._hybrid_act_dist(
                    next_actions[self.d_key], next_actions[self.c_key], prop_ids, prop_params
                )
                next_q_values = next_q_values + self.demo_aux_weight * act_dist.unsqueeze(1)

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            last_critic_loss = float(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actor_actions = self._deterministic_action(self.actor, replay_data.observations)
            q_values = self.critic(replay_data.observations, actor_actions)[0]

            prop_ids, prop_params = self._compute_propagated_actions(
                replay_data.observations, demo_obs_t, demo_ids, demo_params
            )
            act_dist = self._hybrid_act_dist(
                actor_actions[self.d_key], actor_actions[self.c_key], prop_ids, prop_params
            )

            actor_loss = -(q_values + self.demo_aux_weight * act_dist.unsqueeze(1)).mean()
            last_actor_loss = float(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                if self.actor_target is not None:
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        if hasattr(self, "logger"):
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/actor_loss", last_actor_loss)
            self.logger.record("train/critic_loss", last_critic_loss)
            self.logger.record("train/demo_aux_weight", self.demo_aux_weight)
            self.logger.record("train/demo_k", self.demo_k)
            self.logger.record("train/demo_id_margin", self.demo_id_margin)