from __future__ import annotations

import copy
import numpy as np
from typing import Optional, Tuple

import torch as th
from torch.nn import functional as F

from stable_baselines3.hsac import HSAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import HybridDictReplayBufferSamples
from stable_baselines3.hsac_dex.demo_buffer import DemoBuffer


class HSAC_DEX(HSAC):
    """DEX-style hybrid actor-critic built on HSAC."""

    def __init__(
        self,
        *args,
        demo_path: Optional[str] = None,
        demo_batch_size: int = 256,
        demo_aux_weight: float = 1.0,
        replay_demo_ratio: float = 0.0,
        demo_k: int = 5,
        demo_id_margin: float = 1.0,
        demo_dist_threshold: float = 2.0,
        **kwargs,
    ):
        self.demo_path = demo_path
        self.demo_batch_size = demo_batch_size
        self.demo_aux_weight = demo_aux_weight
        self.replay_demo_ratio = replay_demo_ratio
        self.demo_k = demo_k
        self.demo_id_margin = demo_id_margin
        self.demo_dist_threshold = demo_dist_threshold
        self._demo_buffer: Optional[DemoBuffer] = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.demo_path is not None:
            # 创建 demo_buffer，使用与 replay_buffer 相同的 n_envs
            self._demo_buffer = DemoBuffer.from_npz(
                path=self.demo_path,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                n_envs=self.n_envs,  # 使用相同的 n_envs
            )
            
            self.demo_batch_size = min(self.demo_batch_size, self._demo_buffer.size())

            # 将 demo_buffer 的数据复制到 replay_buffer
            # 现在两者的 n_envs 一致，可以直接复制内部数组
            print(f"\nCopying demo data to replay buffer...")
            print(f"  Demo buffer size: {self._demo_buffer.size()}, pos: {self._demo_buffer.pos}")
            print(f"  Replay buffer capacity: {self.replay_buffer.buffer_size}")
            
            demo_size = self._demo_buffer.pos  # 实际填充的位置
            
            # 确保 replay_buffer 有足够空间
            if demo_size > self.replay_buffer.buffer_size:
                print(f"Warning: Demo size ({demo_size}) exceeds replay buffer size ({self.replay_buffer.buffer_size})")
                demo_size = self.replay_buffer.buffer_size
            
            # 直接复制内部数组（高效）
            for key in self.replay_buffer.observations.keys():
                self.replay_buffer.observations[key][:demo_size] = \
                    self._demo_buffer.observations[key][:demo_size].copy()
                self.replay_buffer.next_observations[key][:demo_size] = \
                    self._demo_buffer.next_observations[key][:demo_size].copy()
            
            for key in self.replay_buffer.actions.keys():
                self.replay_buffer.actions[key][:demo_size] = \
                    self._demo_buffer.actions[key][:demo_size].copy()
            
            self.replay_buffer.rewards[:demo_size] = self._demo_buffer.rewards[:demo_size].copy()
            self.replay_buffer.dones[:demo_size] = self._demo_buffer.dones[:demo_size].copy()
            
            if self.replay_buffer.handle_timeout_termination:
                self.replay_buffer.timeouts[:demo_size] = 0.0
            
            # 更新 replay_buffer 的位置
            self.replay_buffer.pos = demo_size
            self.replay_buffer.full = (demo_size >= self.replay_buffer.buffer_size)
            
            print(f"✓ Successfully copied {demo_size} demo transitions to replay buffer")
            print(f"  Replay buffer now: pos={self.replay_buffer.pos}, size={self.replay_buffer.size()}\n")


    def _compute_propagated_actions(
        self,
        obs: th.Tensor,
        obs_demo: th.Tensor,
        demo_ids: th.Tensor,
        demo_params: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # Ensure 2D feature tensors so cdist/topk stays consistent
        if obs.dim() > 2:
            obs = obs.reshape(obs.shape[0], -1)
        if obs_demo.dim() > 2:
            obs_demo = obs_demo.reshape(obs_demo.shape[0], -1)

        # Flatten demo id/params to match demo batch dimension
        if demo_ids.dim() > 1:
            demo_ids = demo_ids.squeeze(-1)
        demo_ids = demo_ids.long()
        if demo_params.dim() > 2:
            demo_params = demo_params.reshape(demo_params.shape[0], -1)

        k = min(self.demo_k, obs_demo.shape[0])
        l2_pair = th.cdist(obs, obs_demo)
        topk_values, topk_indices = l2_pair.topk(k, dim=1, largest=False)
        topk_weights = F.softmax(-topk_values, dim=1)

        # Compute validity mask: max distance among k nearest neighbors < threshold
        # topk_values: (batch, k), take max along k dimension
        max_topk_dist = topk_values.max(dim=1)[0]  # (batch,)
        valid_mask = max_topk_dist < self.demo_dist_threshold  # (batch,)

        #debug (convert to numpy)
        # try:
        #     # 选择要打印的obs数量
        #     num_obs_to_print = min(3, obs.shape[0])

        #     # 转换为numpy
        #     obs_np = obs.cpu().numpy()
        #     demo_obs_np = obs_demo.cpu().numpy()
        #     topk_indices_np = topk_indices.cpu().numpy()

        #     for i in range(num_obs_to_print):
        #         nearest_demo_indices = topk_indices_np[i]
        #         nearest_demo_obs = demo_obs_np[nearest_dWemo_indices]
        #         dist_diff = obs_np[i] - nearest_demo_obs

        #     print(dist_diff)

        # except Exception as e:
        #     print(f"Debug output failed: {e}")
        

        topk_ids = demo_ids[topk_indices]  # (batch, k)
        topk_params = demo_params[topk_indices]  # (batch, k, param_dim)

        # batch_size = obs.shape[0]
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

        return prop_ids, prop_params, topk_values, valid_mask

    def _hybrid_act_dist(
        self,
        pred_ids: th.Tensor,
        pred_params: th.Tensor,
        target_ids: th.Tensor,
        target_params: th.Tensor,
    ) -> th.Tensor:
        """
        计算混合动作距离 d((a,x), (a_target, x_target)).
        
        支持两种模式:
        1. 单对单: pred_params (batch, param_dim), target_params (batch, param_dim)
           返回 (batch,)
        2. 多对一: pred_params (batch, n_actions, param_dim), target_params (batch, param_dim)
           返回 (batch, n_actions)

        self.demo_id_margin is the upper bound for the param distance.
        """
        # 判断是单对单还是多对一模式
        if pred_params.dim() == 2:  # (batch, param_dim)
            # 单对单模式: 计算标准欧式距离
            param_dist = th.sqrt((pred_params - target_params).pow(2).sum(dim=1))
            id_mismatch = (pred_ids != target_ids).float() * self.demo_id_margin
        else:  # pred_params.dim() == 3: (batch, n_actions, param_dim)
            # 多对一模式: 为每个离散动作计算标准欧式距离
            # target_params: (batch, param_dim) -> (batch, 1, param_dim)
            param_diff = pred_params - target_params.unsqueeze(1)
            param_dist = th.sqrt((param_diff ** 2).sum(dim=2))  # (batch, n_actions)
            
            # pred_ids: (batch, n_actions), target_ids: (batch,) -> (batch, 1)
            id_mismatch = (pred_ids != target_ids.unsqueeze(1)).float() * self.demo_id_margin

        return th.where(id_mismatch > 0, id_mismatch, param_dist)

    def _concat_replay_samples(
        self,
        first: HybridDictReplayBufferSamples,
        second: HybridDictReplayBufferSamples,
    ) -> HybridDictReplayBufferSamples:
        observations = {
            key: th.cat([first.observations[key], second.observations[key]], dim=0)
            for key in first.observations.keys()
        }
        actions = {
            key: th.cat([first.actions[key], second.actions[key]], dim=0)
            for key in first.actions.keys()
        }
        next_observations = {
            key: th.cat([first.next_observations[key], second.next_observations[key]], dim=0)
            for key in first.next_observations.keys()
        }
        dones = th.cat([first.dones, second.dones], dim=0)
        rewards = th.cat([first.rewards, second.rewards], dim=0)
        return HybridDictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

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
        
        #debug
        dist_losses = []
        topk_dist_means = []
        valid_ratios = []
        max_dists = []

        # 使用 self._total_timesteps 作为归一化上限，确保线性衰减到0
        total_timesteps = getattr(self, '_total_timesteps', None)
        if total_timesteps is None or total_timesteps <= 0:
            total_timesteps = 1e7  # fallback，防止未设置
        # 以当前已采样环境步数 self.num_timesteps 归一化
        decay_coef = max(0.0, 1.0 - float(self.num_timesteps) / float(total_timesteps))
        demo_aux_weight_scaled = self.demo_aux_weight * decay_coef

        for gradient_step in range(gradient_steps):
            demo_ratio = float(np.clip(self.replay_demo_ratio, 0.0, 1.0))
            demo_replay_batch = int(round(batch_size * demo_ratio))
            replay_batch = batch_size - demo_replay_batch

            if demo_replay_batch > 0 and self._demo_buffer is None:
                raise RuntimeError("demo_path must be provided when replay_demo_ratio > 0")

            if replay_batch > 0:
                replay_part = self.replay_buffer.sample(replay_batch, env=self._vec_normalize_env)  # type: ignore[union-attr]
            if demo_replay_batch > 0:
                demo_part = self._demo_buffer.sample(demo_replay_batch, env=self._vec_normalize_env)

            if replay_batch == 0:
                replay_data = demo_part
            elif demo_replay_batch == 0:
                replay_data = replay_part
            else:
                replay_data = self._concat_replay_samples(replay_part, demo_part)

            if self._demo_buffer is None:
                raise RuntimeError("demo_path must be provided for HSAC_DEX")

            # Sample from demo buffer (returns HybridDictReplayBufferSamples)
            demo_data = self._demo_buffer.sample(self.demo_batch_size, env=self._vec_normalize_env)
            
            # Extract observation tensor (use 'observation' key from dict)
            demo_obs_t = demo_data.observations["observation"]
            
            # Extract action components
            demo_ids = demo_data.actions["discrete"]
            demo_params = demo_data.actions["continuous"]

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

                prop_ids, prop_params, _, valid_mask = self._compute_propagated_actions(
                    replay_data.next_observations["observation"], demo_obs_t, demo_ids, demo_params
                )
                act_dist = self._hybrid_act_dist(
                    next_actions[self.d_key], next_actions[self.c_key], prop_ids, prop_params
                )
                if demo_aux_weight_scaled > 0:
                    # Apply mask: only penalize similar states
                    masked_dist = act_dist * valid_mask.float()
                    next_q_values = next_q_values - demo_aux_weight_scaled * masked_dist.unsqueeze(1)

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

            # 计算传播的演示动作: (a^e, x^e)
            prop_ids, prop_params, topk_values, valid_mask = self._compute_propagated_actions(
                replay_data.observations["observation"], demo_obs_t, demo_ids, demo_params
            )
            topk_dist_means.append(topk_values.mean().item())
            valid_ratios.append(valid_mask.float().mean().item())
            max_dists.append(topk_values.max(dim=1)[0].mean().item())
            
            # 为每个离散动作a计算 d((a,x̃), (a^e,x^e))
            # all_continuous_actions: (batch, n_actions, param_dim)
            # 创建所有离散动作的索引: (batch, n_actions)
            batch_size = all_continuous_actions.shape[0]
            n_actions = all_continuous_actions.shape[1]
            all_discrete_actions = th.arange(n_actions, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # 使用 _hybrid_act_dist 计算距离 (batch, n_actions)
            act_dist_all = self._hybrid_act_dist(
                all_discrete_actions, all_continuous_actions, prop_ids, prop_params
            )
            
            # Apply mask: (batch, n_actions) * (batch, 1) -> (batch, n_actions)
            masked_dist_all = act_dist_all * valid_mask.unsqueeze(1).float()

            # 计算加权损失: L_π = E[Σ_a π_tsk(a|s)[...]]
            weighted_loss = task_probs * (
                ent_coef_task * task_log_probs +
                ent_coef_param * all_continuous_log_probs +
                demo_aux_weight_scaled * masked_dist_all -
                all_q_values
            )
            # Σ_a: 对所有离散动作求和; E: 对batch求均值
            actor_loss = weighted_loss.sum(dim=1).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            #debug
            dist_loss = (task_probs * demo_aux_weight_scaled * masked_dist_all).sum(dim=1).mean()
            dist_losses.append(dist_loss.item())

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
            self.logger.record("train/demo_aux_weight", demo_aux_weight_scaled)
            self.logger.record("train/dist_loss", float(np.mean(dist_losses)) if dist_losses else 0.0)
            self.logger.record("train/topk_dist_mean", float(np.mean(topk_dist_means)) if topk_dist_means else 0.0)
            self.logger.record("train/valid_demo_ratio", float(np.mean(valid_ratios)) if valid_ratios else 0.0)
            self.logger.record("train/max_topk_dist", float(np.mean(max_dists)) if max_dists else 0.0)
            # self.logger.record("train/demo_k", self.demo_k)
            # self.logger.record("train/demo_id_margin", self.demo_id_margin)