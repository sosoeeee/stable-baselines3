from __future__ import annotations

import copy
import numpy as np
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
        demo_norm: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        k = min(self.demo_k, obs_demo.shape[0])
        obs_norm = (obs ** 2).sum(dim=1, keepdim=True)
        demo_norm = demo_norm if demo_norm is not None else (obs_demo ** 2).sum(dim=1, keepdim=True).T
        l2_pair = obs_norm + demo_norm - 2.0 * (obs @ obs_demo.T)
        topk_values, topk_indices = l2_pair.topk(k, dim=1, largest=False)
        topk_weights = F.softmax(-topk_values, dim=1)

        topk_ids = demo_ids[topk_indices]  # (batch, k)
        topk_params = demo_params[topk_indices]  # (batch, k, param_dim)

        batch_size = obs.shape[0]
        n_actions = self.actor.n_discrete_actions
        id_weight = (
            F.one_hot(topk_ids, num_classes=n_actions).float() * topk_weights.unsqueeze(-1)
        ).sum(dim=1)
        prop_ids = th.argmax(id_weight, dim=1)

        mask = topk_ids == prop_ids.unsqueeze(1)
        weights_masked = topk_weights * mask
        weights_sum = weights_masked.sum(dim=1, keepdim=True)
        use_mask = weights_sum > 0
        weights = th.where(use_mask, weights_masked, topk_weights)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        prop_params = (weights.unsqueeze(-1) * topk_params).sum(dim=1)

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
        # Update optimizers learning rate (actor/critic + entropy coeffs)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_task_optimizer is not None:
            optimizers.append(self.ent_coef_task_optimizer)
        if self.ent_coef_param_optimizer is not None:
            optimizers.append(self.ent_coef_param_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_task_losses, ent_coef_param_losses = [], []
        ent_coefs_task, ent_coefs_param = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            if self._demo_buffer is None:
                raise RuntimeError("demo_path must be provided for HSAC_DEX")
            demo_obs, demo_actions = self._demo_buffer.sample(self.demo_batch_size)
            # If environment observations are normalized (VecNormalize), apply same normalization
            if getattr(self, "_vec_normalize_env", None) is not None:
                # normalize_obs expects numpy arrays and does not update running stats
                demo_obs = self._vec_normalize_env.normalize_obs(demo_obs)
            demo_obs_t = th.as_tensor(demo_obs, device=self.device, dtype=th.float32)
            demo_ids = th.as_tensor(demo_actions["id"], device=self.device).long().flatten()
            demo_params = th.as_tensor(demo_actions["params"], device=self.device, dtype=th.float32)
            demo_norm = (demo_obs_t ** 2).sum(dim=1, keepdim=True).T

            # Current actions and log probs from actor (for entropy terms)
            actions_pi, discrete_log_prob, continuous_log_prob = self.actor.action_log_prob(replay_data.observations)

            # Get entropy coefficients
            if self.ent_coef_task_optimizer is not None and self.log_ent_coef_task is not None:
                ent_coef_task = th.exp(self.log_ent_coef_task.detach())
                ent_coef_task_loss = -(self.log_ent_coef_task * (discrete_log_prob + self.target_entropy_task).detach()).mean()
                ent_coef_task_losses.append(ent_coef_task_loss.item())
            else:
                ent_coef_task = self.ent_coef_task_tensor
                ent_coef_task_loss = None

            if self.ent_coef_param_optimizer is not None and self.log_ent_coef_param is not None:
                ent_coef_param = th.exp(self.log_ent_coef_param.detach())
                ent_coef_param_loss = -(self.log_ent_coef_param * (continuous_log_prob + self.target_entropy_param).detach()).mean()
                ent_coef_param_losses.append(ent_coef_param_loss.item())
            else:
                ent_coef_param = self.ent_coef_param_tensor
                ent_coef_param_loss = None

            ent_coefs_task.append(ent_coef_task.item())
            ent_coefs_param.append(ent_coef_param.item())

            # Optimize entropy coefficients
            if ent_coef_task_loss is not None and self.ent_coef_task_optimizer is not None:
                self.ent_coef_task_optimizer.zero_grad()
                ent_coef_task_loss.backward()
                self.ent_coef_task_optimizer.step()

            if ent_coef_param_loss is not None and self.ent_coef_param_optimizer is not None:
                self.ent_coef_param_optimizer.zero_grad()
                ent_coef_param_loss.backward()
                self.ent_coef_param_optimizer.step()

            with th.no_grad():
                next_actions, next_discrete_log_prob, next_continuous_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                next_q_values = next_q_values - ent_coef_task * next_discrete_log_prob.reshape(-1, 1)
                next_q_values = next_q_values - ent_coef_param * next_continuous_log_prob.reshape(-1, 1)

                next_actions_det = self._deterministic_action(self.actor_target, replay_data.next_observations)

                prop_ids, prop_params = self._compute_propagated_actions(
                    replay_data.next_observations, demo_obs_t, demo_ids, demo_params, demo_norm
                )
                act_dist = self._hybrid_act_dist(
                    next_actions_det[self.d_key], next_actions_det[self.c_key], prop_ids, prop_params
                )
                if self.demo_aux_weight > 0:
                    next_q_values = next_q_values + self.demo_aux_weight * act_dist.unsqueeze(1)

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            logits = self.actor.get_task_dist_params(replay_data.observations)
            task_dist = th.distributions.Categorical(logits=logits)
            task_probs = task_dist.probs
            task_log_probs = th.log_softmax(logits, dim=-1)

            all_means, all_log_stds = self.actor.get_all_param_dist_params(replay_data.observations)
            all_stds = th.exp(all_log_stds)
            noise = th.randn_like(all_means)
            all_continuous_pretanh = all_means + all_stds * noise
            all_continuous_actions = th.tanh(all_continuous_pretanh)

            gaussian_log_prob = -0.5 * (
                ((all_continuous_pretanh - all_means) / (all_stds + 1e-6)) ** 2
                + 2 * all_log_stds
                + np.log(2 * np.pi)
            )
            gaussian_log_prob = gaussian_log_prob.sum(dim=-1)
            squash_correction = th.log(1 - all_continuous_actions ** 2 + 1e-6).sum(dim=-1)
            all_continuous_log_probs = gaussian_log_prob - squash_correction

            all_q_values = self.critic.q1_forward_all_discrete(replay_data.observations, all_continuous_actions)
            all_q_values = all_q_values.squeeze(-1)

            with th.no_grad():
                actor_actions_det = self._deterministic_action(self.actor, replay_data.observations)
                prop_ids, prop_params = self._compute_propagated_actions(
                    replay_data.observations, demo_obs_t, demo_ids, demo_params, demo_norm
                )
                act_dist = self._hybrid_act_dist(
                    actor_actions_det[self.d_key], actor_actions_det[self.c_key], prop_ids, prop_params
                )

            shaping = self.demo_aux_weight * act_dist.unsqueeze(1) if self.demo_aux_weight > 0 else 0.0
            weighted_loss = task_probs * (
                ent_coef_task * task_log_probs +
                ent_coef_param * all_continuous_log_probs -
                (all_q_values + shaping)
            )
            actor_loss = weighted_loss.sum(dim=1).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                if self.actor_target is not None:
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        if hasattr(self, "logger"):
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef_task", float(np.mean(ent_coefs_task)) if ent_coefs_task else 0.0)
            self.logger.record("train/ent_coef_param", float(np.mean(ent_coefs_param)) if ent_coefs_param else 0.0)
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)) if actor_losses else 0.0)
            self.logger.record("train/critic_loss", float(np.mean(critic_losses)) if critic_losses else 0.0)
            if len(ent_coef_task_losses) > 0:
                self.logger.record("train/ent_coef_task_loss", float(np.mean(ent_coef_task_losses)))
            if len(ent_coef_param_losses) > 0:
                self.logger.record("train/ent_coef_param_loss", float(np.mean(ent_coef_param_losses)))
            self.logger.record("train/demo_aux_weight", self.demo_aux_weight)
            self.logger.record("train/demo_k", self.demo_k)
            self.logger.record("train/demo_id_margin", self.demo_id_margin)
