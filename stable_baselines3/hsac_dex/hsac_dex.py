from __future__ import annotations

import copy
import numpy as np
from typing import Optional, Tuple

import torch as th
from torch.nn import functional as F
import time

from stable_baselines3.hsac import HSAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import HybridDictReplayBufferSamples
from stable_baselines3.hsac_dex.demo_buffer import DemoBuffer
from stable_baselines3.common.vec_env import unwrap_vec_normalize


class HSAC_DEX(HSAC):
    """DEX-style hybrid actor-critic built on HSAC."""

    def __init__(
        self,
        *args,
        demo_path: Optional[str] = None,
        demo_batch_size: int = 256,
        demo_aux_weight: float = 1.0,
        use_dex_aux_loss: bool = True,
        use_timing_profile: bool = False,
        demo_bc_weight: float = 1.0,
        use_bc_loss: bool = True,
        replay_demo_ratio: float = 0.0,
        demo_k: int = 5,
        demo_id_margin: float = 1.0,
        demo_dist_threshold: float = 2.0,
        demo_buffer_kwargs: Optional[dict] = None,
        entropy_ema_alpha: float = 0.1,             # EMA平滑系数，越小越平滑
        entropy_change_threshold: float = 0.002,    # 熵变化率阈值，超过此值认为"显著上升"
        entropy_update_interval: int = 10,          # 每隔多少次train()调用更新一次EMA
        max_entropy_weighted: float = 2.0,          # 熵值加权项的最大值：entropy * ent_coef_task
        target_q_clip: Optional[float] = None,      # 目标Q值截断范围，None表示不截断
        **kwargs,
    ):
        self.demo_path = demo_path
        self.demo_batch_size = demo_batch_size
        self.demo_aux_weight = demo_aux_weight
        self.use_dex_aux_loss = use_dex_aux_loss
        self.use_timing_profile = use_timing_profile
        self.demo_bc_weight = demo_bc_weight
        self.use_bc_loss = use_bc_loss
        self.replay_demo_ratio = replay_demo_ratio
        self.demo_k = demo_k
        self.demo_id_margin = demo_id_margin
        self.demo_dist_threshold = demo_dist_threshold
        self.demo_buffer_kwargs = demo_buffer_kwargs or {}
        self._demo_buffer: Optional[DemoBuffer] = None
        # control the entropy coef of discrete action
        self.max_entropy_weighted = max_entropy_weighted
        self.entropy_ema_alpha = entropy_ema_alpha
        self.entropy_change_threshold = entropy_change_threshold
        self.entropy_update_interval = entropy_update_interval
        self.task_entropy_ema: Optional[float] = None
        self._train_call_count: int = 0  # 记录train()调用次数
        self._prev_task_entropy_ema: Optional[float] = None
        self._task_entropy_change_rate: Optional[float] = 0.0
        self._accumulated_task_entropy: float = 0.0  # 累积的熵值
        self._accumulated_entropy_samples: int = 0  # 累积的样本数
        self._timing_snapshot: Optional[dict] = None
        self.target_q_clip = target_q_clip
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        
        self.task_entropy_ema = float(np.log(self.n_discrete_actions)) if hasattr(self, 'target_entropy_task') else 0.0
        
        if self.demo_path is not None:
            demo_buffer_kwargs = dict(self.demo_buffer_kwargs)
            # HybridHerReplayBuffer requires an env to compute HER rewards.
            demo_buffer_kwargs.setdefault("env", self.env)
            # 创建 demo_buffer，使用独立的单环境配置
            self._demo_buffer = DemoBuffer.from_npz(
                path=self.demo_path,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                **demo_buffer_kwargs,
            )
            
            self.demo_batch_size = min(self.demo_batch_size, self._demo_buffer.size())

            demo_size = self._demo_buffer.pos
            
            if demo_size == 0:
                print("Warning: No demo data to load into replay buffer.")
            else:
                # Update VecNormalize statistics with demonstration data if environment is wrapped
                vec_normalize = unwrap_vec_normalize(self.env)
                if vec_normalize is not None:
                    print("Updating VecNormalize statistics with demonstration data...")

                    # Prepare demonstration observations for updating statistics
                    if vec_normalize.norm_obs:
                        demo_obs = {
                            key: self._demo_buffer.observations[key][:demo_size].copy()
                            for key in self._demo_buffer.observations
                        }
                        vec_normalize.update_from_data(observations=demo_obs, rewards=None)
                        print(f"  ✓ Updated observation statistics from {demo_size} demo transitions")

                    # Prepare demonstration rewards for updating statistics
                    if vec_normalize.norm_reward:
                        demo_rewards = self._demo_buffer.rewards[:demo_size].copy()
                        vec_normalize.update_from_data(observations=None, rewards=demo_rewards)
                        print(f"  ✓ Updated reward statistics from {demo_size} demo transitions")

                    print("✓ VecNormalize statistics updated with demonstration data\n")


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
        #
        #     # 转换为numpy
        #     obs_np = obs.cpu().numpy()
        #     demo_obs_np = obs_demo.cpu().numpy()
        #     topk_indices_np = topk_indices.cpu().numpy()
        #
        #     for i in range(num_obs_to_print):
        #         nearest_demo_indices = topk_indices_np[i]
        #         nearest_demo_obs = demo_obs_np[nearest_demo_indices]
        #         dist_diff = obs_np[i] - nearest_demo_obs
        #
        #     print(dist_diff)
        #
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
            param_dist = (pred_params - target_params).pow(2).mean(dim=1)
            id_mismatch = (pred_ids != target_ids).float() * self.demo_id_margin
        else:
            # 多对一模式: 为每个离散动作计算标准欧式距离
            # target_params: (batch, param_dim) -> (batch, 1, param_dim)
            param_diff = pred_params - target_params.unsqueeze(1)
            param_dist = (param_diff ** 2).mean(dim=2)  # (batch, n_actions)
            # pred_ids: (batch, n_actions), target_ids: (batch,) -> (batch, 1)
            id_mismatch = (pred_ids != target_ids.unsqueeze(1)).float() * self.demo_id_margin
        
        # 强制 param_dist 的上限为 demo_id_margin
        # 这样选对动作的惩罚最多等于选错动作，永远不会因为连续参数太差而导致选错更“划算”
        param_dist_clamped = th.clamp(param_dist, max=self.demo_id_margin)

        return th.where(id_mismatch > 0, id_mismatch, param_dist_clamped)

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
        # timing accumulators
        sample_times, forward_times, critic_times, actor_times = [], [], [], []
        target_times, entropy_times, dex_times, bc_times, polyak_times = [], [], [], [], []
        bc_losses = []
        
        # debug
        dist_losses = []
        topk_dist_means = []
        valid_ratios = []
        max_dists = []
        task_entropys = []
        critic_grad_norms = []
        actor_grad_norms = []
        
        # Q值统计 debug
        target_q_means = []
        current_q_means = []
        target_q_re_means = []
        target_q_Tent_means = []
        target_q_Pent_means = []
        target_q_dex_mins = []

        # Reward统计 debug
        reward_maxs = []
        
        # 梯度诊断统计 debug
        max_td_error_list = []
        mean_td_error_list = []
        max_obs_val_list = []
        max_desired_goal_val_list = []
        max_achieved_goal_val_list = []
        gradient_spike_count = 0
        
        # 使用 self._total_timesteps 作为归一化上限，确保线性衰减到0
        # 以当前已采样环境步数 self.num_timesteps 归一化
        decay_coef = max(0.0, 1.0 - float(self.num_timesteps) / float(self._total_timesteps))
        demo_aux_weight_scaled = self.demo_aux_weight * decay_coef

        for gradient_step in range(gradient_steps):
            demo_ratio = float(np.clip(self.replay_demo_ratio, 0.0, 1.0))
            demo_replay_batch = round(batch_size * demo_ratio)
            replay_batch = batch_size - demo_replay_batch

            if demo_replay_batch > 0 and self._demo_buffer is None:
                raise RuntimeError("demo_path must be provided when replay_demo_ratio > 0")

            # sampling phase
            if self.use_timing_profile:
                sample_start = time.perf_counter()
            if replay_batch > 0:
                replay_part = self.replay_buffer.sample(replay_batch, env=self._vec_normalize_env)  # type: ignore[union-attr]
            if demo_replay_batch > 0:
                demo_part = self._demo_buffer.sample(demo_replay_batch, env=self._vec_normalize_env)
            if self.use_timing_profile:
                sample_times.append(time.perf_counter() - sample_start)

            if replay_batch == 0:
                replay_data = demo_part
            elif demo_replay_batch == 0:
                replay_data = replay_part
            else:
                replay_data = self._concat_replay_samples(replay_part, demo_part)

            demo_obs_t = None
            demo_ids = None
            demo_params = None
            if self.use_dex_aux_loss:
                if self._demo_buffer is None:
                    raise RuntimeError("demo_path must be provided when use_dex_aux_loss is True")
                # Sample from demo buffer (returns HybridDictReplayBufferSamples)
                demo_data = self._demo_buffer.sample(self.demo_batch_size, env=self._vec_normalize_env)

                # Extract observation tensor (use 'observation' key from dict)
                demo_obs_t = demo_data.observations["observation"]

                # Extract action components
                demo_ids = demo_data.actions["discrete"]
                demo_params = demo_data.actions["continuous"]

            # Single forward pass: Use evaluate_all_actions for current state
            # This computes task distribution, all continuous actions, and log probs in one pass
            if self.use_timing_profile:
                forward_start = time.perf_counter()
            (task_probs, task_log_probs, task_entropy, 
             all_continuous_actions, all_continuous_log_probs) = self.actor.evaluate_all_actions(
                replay_data.observations
            )
            if self.use_timing_profile:
                forward_times.append(time.perf_counter() - forward_start)
            
            task_entropy_mean = task_entropy.mean().item()
            task_entropys.append(task_entropy_mean)
            
            # 累积熵值，用于后续更新EMA
            self._accumulated_task_entropy += task_entropy_mean
            self._accumulated_entropy_samples += 1

            # Timing: entropy coefficient updates
            ent_coef_start = None
            if self.use_timing_profile:
                ent_coef_start = time.perf_counter()

            if self.ent_coef_task_optimizer is not None and self.log_ent_coef_task is not None:
                ent_coef_task = th.exp(self.log_ent_coef_task.detach())
                
                # 动态截断逻辑：如果 entropy * ent_coef_task > max_entropy_weighted，则截断 ent_coef_task
                entropy_weighted = task_entropy_mean * ent_coef_task.item()
                if entropy_weighted > self.max_entropy_weighted:
                    # 计算需要的最大系数值: max_ent_coef = max_entropy_weighted / entropy
                    max_allowed_coef = self.max_entropy_weighted / max(task_entropy_mean, 1e-6)
                    ent_coef_task = th.clamp(ent_coef_task, max=max_allowed_coef)
                
                # Use exact entropy instead of sampled log prob
                ent_coef_task_loss = self.log_ent_coef_task * ((task_entropy - self.target_entropy_task).detach()).mean()
                ent_coef_task_losses.append(ent_coef_task_loss.item())

                # 检查是否应该跳过优化：
                task_below_target = task_entropy_mean < self.target_entropy_task
                task_above_target = task_entropy_mean > self.target_entropy_task
                task_rising = self._task_entropy_change_rate > self.entropy_change_threshold
                task_falling = self._task_entropy_change_rate < -self.entropy_change_threshold

                skip_task_update = (task_below_target and task_rising) or (task_above_target and task_falling)
                
                # 如果已经达到动态截断值，也跳过优化
                if entropy_weighted >= self.max_entropy_weighted and task_below_target:
                    skip_task_update = True
                
                if skip_task_update:
                    ent_coef_task_loss = None  # 跳过此次更新
            else:
                ent_coef_task = self.ent_coef_task_tensor
                ent_coef_task_loss = None

            if self.ent_coef_param_optimizer is not None and self.log_ent_coef_param is not None:
                ent_coef_param = th.exp(self.log_ent_coef_param.detach())
                # Get discrete actions from replay buffer for action-specific target entropy
                discrete_actions = replay_data.actions[self.d_key].long().squeeze(-1)
                current_target_entropy_param = self.target_entropy_param[discrete_actions]
                # Get continuous log prob for the sampled discrete action from all_continuous_log_probs
                sampled_continuous_log_prob = all_continuous_log_probs.gather(
                    1, discrete_actions.unsqueeze(-1)
                ).squeeze(-1)  # (batch_size,)
                ent_coef_param_loss = -(
                    self.log_ent_coef_param * (
                        sampled_continuous_log_prob + current_target_entropy_param
                    ).detach()
                ).mean()
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

            if self.use_timing_profile and ent_coef_start is not None:
                entropy_times.append(time.perf_counter() - ent_coef_start)

            target_start = None
            if self.use_timing_profile:
                target_start = time.perf_counter()

            with th.no_grad():
                # Use evaluate_all_actions for next state (with param_mask applied)
                (next_task_probs, next_task_log_probs, _, 
                 all_next_continuous_actions, all_next_continuous_log_probs) = self.actor.evaluate_all_actions(
                    replay_data.next_observations
                )
                
                # Compute min Q-values across all critics for all discrete actions
                # all_next_q_values: (batch_size, n_discrete_actions)
                all_next_q_values = self.critic_target.forward_all_discrete(
                    replay_data.next_observations, all_next_continuous_actions
                )
                if self.use_dex_aux_loss:
                    # ------------ DEX: Compute demo distance for all discrete actions ------------
                    prop_ids, prop_params, _, valid_mask = self._compute_propagated_actions(
                        replay_data.next_observations["observation"], demo_obs_t, demo_ids, demo_params
                    )

                    # Create all discrete action indices: (batch_size, n_discrete_actions)
                    batch_size_next = all_next_continuous_actions.shape[0]
                    n_actions_next = all_next_continuous_actions.shape[1]
                    all_next_discrete_actions = th.arange(n_actions_next, device=self.device).unsqueeze(0).expand(batch_size_next, -1)

                    # Compute demo distance for all discrete actions: (batch_size, n_discrete_actions)
                    all_next_act_dist = self._hybrid_act_dist(
                        all_next_discrete_actions, all_next_continuous_actions, prop_ids, prop_params
                    )

                    # Apply mask: (batch_size, n_discrete_actions) * (batch_size, 1)
                    all_next_masked_dist = all_next_act_dist * valid_mask.unsqueeze(1).float()
                    dex = (-next_task_probs * demo_aux_weight_scaled * all_next_masked_dist).sum(dim=1, keepdim=True)
                else:
                    all_next_masked_dist = th.zeros_like(all_next_q_values)
                    dex = th.zeros((all_next_q_values.shape[0], 1), device=self.device)
                # -------------------------------------------------------------------------------
                
                # Compute value for each discrete action: Q - alpha_task*log(pi_task) - alpha_param*log(pi_param) - lambda*d
                # (batch_size, n_discrete_actions)
                all_next_values = (
                    all_next_q_values 
                    - ent_coef_task * next_task_log_probs 
                    - ent_coef_param * all_next_continuous_log_probs
                    - demo_aux_weight_scaled * all_next_masked_dist
                )
                
                # Compute expected value by summing over discrete actions weighted by task policy
                # next_v = sum_a' [ pi_task(a'|s') * V(s', a') ]
                next_v = (next_task_probs * all_next_values).sum(dim=1, keepdim=True)  # (batch_size, 1)

                # debug
                Tent = (-next_task_probs * ent_coef_task * next_task_log_probs).sum(dim=1, keepdim=True)
                Pent = (-next_task_probs * ent_coef_param * all_next_continuous_log_probs).sum(dim=1, keepdim=True)

                # Compute target Q values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v
                
                # Apply clipping to target Q values if specified
                if self.target_q_clip is not None:
                    target_q_values = th.clamp(target_q_values, -self.target_q_clip, self.target_q_clip)

                # debug
                target_q_re_means.append(replay_data.rewards.mean().item())
                target_q_Tent_means.append(Tent.mean().item())
                target_q_Pent_means.append(Pent.mean().item())
                target_q_dex_mins.append(dex.min().item())

            if self.use_timing_profile and target_start is not None:
                target_times.append(time.perf_counter() - target_start)

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            # change to Huber Loss (less sensitive to outliers that HER creates)
            # critic_loss = sum(F.smooth_l1_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            
            # debug
            # 记录 target Q 和 current Q 的均值
            target_q_means.append(target_q_values.mean().item())
            # 使用第一个 Q 网络的输出作为代表（或者可以用所有 Q 网络的平均值）
            current_q_means.append(current_q_values[0].mean().item())
            # 记录当前 batch 的 reward 最大值
            reward_maxs.append(replay_data.rewards.max().item())

            if self.use_timing_profile:
                critic_start = time.perf_counter()
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            
            # debug
            # --- CRITIC EXPLOSION DIAGNOSTIC BLOCK ---
            # 1. Calculate the raw gradient norm manually
            grad_norm = 0.0
            for p in self.critic.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            critic_grad_norms.append(grad_norm)

            # 2. Trigger diagnostic only if the gradient is exploding
            if grad_norm > 500.0:
                gradient_spike_count += 1
                
                with th.no_grad():
                    # Hypothesis A: The Outlier Illusion
                    max_td_errors = []
                    mean_td_errors = []
                    for current_q in current_q_values:
                        abs_td_error = th.abs(current_q - target_q_values)
                        max_td_errors.append(abs_td_error.max().item())
                        mean_td_errors.append(abs_td_error.mean().item())
                    
                    max_td = max(max_td_errors)
                    mean_td = sum(mean_td_errors) / len(mean_td_errors)
                    max_td_error_list.append(max_td)
                    mean_td_error_list.append(mean_td)

                    # Hypothesis B: Unnormalized Inputs (Goals)
                    obs = replay_data.observations
                    max_obs_val_list.append(obs['observation'].abs().max().item())
                    max_desired_goal_val_list.append(obs['desired_goal'].abs().max().item())
                    max_achieved_goal_val_list.append(obs['achieved_goal'].abs().max().item())
            # -----------------------------------------
            
            # Clip gradients
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            
            self.critic.optimizer.step()
            if self.use_timing_profile:
                critic_times.append(time.perf_counter() - critic_start)

            # Actor loss uses evaluate_all_actions results computed above
            # No need to re-evaluate since entropy coef losses used .detach()

            # Compute Q-values for all discrete actions using min over all critics
            all_q_values = self.critic.forward_all_discrete(replay_data.observations, all_continuous_actions)

            dex_start = None
            if self.use_timing_profile and self.use_dex_aux_loss:
                dex_start = time.perf_counter()

            if self.use_dex_aux_loss:
                # 计算传播的演示动作: (a^e, x^e)
                prop_ids, prop_params, topk_values, valid_mask = self._compute_propagated_actions(
                    replay_data.observations["observation"], demo_obs_t, demo_ids, demo_params
                )
                topk_dist_means.append(topk_values.mean().item())
                valid_ratios.append(valid_mask.float().mean().item())
                max_dists.append(topk_values.max(dim=1)[0].max().item())

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
            else:
                masked_dist_all = th.zeros_like(all_q_values)

            if self.use_timing_profile and dex_start is not None:
                dex_times.append(time.perf_counter() - dex_start)

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

            bc_loss = th.tensor(0.0, device=self.device)
            bc_start = None
            if self.use_timing_profile and self.use_bc_loss and demo_replay_batch > 0:
                bc_start = time.perf_counter()

            if self.use_bc_loss and demo_replay_batch > 0:
                if replay_batch > 0:
                    demo_slice = slice(replay_batch, replay_batch + demo_replay_batch)
                else:
                    demo_slice = slice(0, demo_replay_batch)

                demo_obs_for_bc = {
                    key: replay_data.observations[key][demo_slice]
                    for key in replay_data.observations.keys()
                }
                demo_target_ids = replay_data.actions[self.d_key][demo_slice].long().view(-1)
                demo_target_params = replay_data.actions[self.c_key][demo_slice]

                # Use logits CE for discrete BC so gradients flow through the task policy.
                demo_task_logits = self.actor.get_task_dist_params(demo_obs_for_bc)
                bc_id_loss = F.cross_entropy(demo_task_logits, demo_target_ids)

                # Regress continuous params conditioned on target discrete action.
                demo_mean_actions, _, _ = self.actor.get_param_dist_params(demo_obs_for_bc, demo_target_ids)
                demo_mask = self.actor.param_mask[demo_target_ids.long()]
                demo_pred_params = th.tanh(demo_mean_actions)
                # Normalize masked MSE by each sample's valid parameter dims to avoid scale bias.
                param_sq_error = (demo_pred_params - demo_target_params).pow(2) * demo_mask
                valid_dims = demo_mask.sum(dim=1).clamp_min(1.0)
                bc_param_loss = (param_sq_error.sum(dim=1) / valid_dims).mean()
                bc_loss = bc_id_loss + bc_param_loss

            if self.use_timing_profile and bc_start is not None:
                bc_times.append(time.perf_counter() - bc_start)

            bc_losses.append(bc_loss.item())
            actor_total_loss = actor_loss + self.demo_bc_weight * bc_loss

            if self.use_timing_profile:
                actor_start = time.perf_counter()
            self.actor.optimizer.zero_grad()
            actor_total_loss.backward()
            
            # Compute gradient norm before clipping
            # debug
            actor_grad_norm_before_clip = 0.0
            for param in self.actor.parameters():
                if param.grad is not None:
                    actor_grad_norm_before_clip += param.grad.data.norm(2).item() ** 2
            actor_grad_norm_before_clip = actor_grad_norm_before_clip ** 0.5
            
            # Clip gradients
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            actor_grad_norms.append(actor_grad_norm_before_clip)
            
            self.actor.optimizer.step()
            if self.use_timing_profile:
                actor_times.append(time.perf_counter() - actor_start)

            if gradient_step % self.target_update_interval == 0:
                polyak_start = None
                if self.use_timing_profile:
                    polyak_start = time.perf_counter()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

                if self.use_timing_profile and polyak_start is not None:
                    polyak_times.append(time.perf_counter() - polyak_start)

        # 每隔 entropy_update_interval 次 train() 调用更新一次 EMA
        self._train_call_count += 1
        avg_task_entropy = self._accumulated_task_entropy / self._accumulated_entropy_samples
        if self.task_entropy_ema is None:
            self.task_entropy_ema = avg_task_entropy
        else:
            self.task_entropy_ema = (1 - self.entropy_ema_alpha) * self.task_entropy_ema + self.entropy_ema_alpha * avg_task_entropy
        self._accumulated_task_entropy = 0.0
        self._accumulated_entropy_samples = 0

        if self._train_call_count >= self.entropy_update_interval:
            if self._prev_task_entropy_ema is not None:
                self._task_entropy_change_rate = self.task_entropy_ema - self._prev_task_entropy_ema
                self._prev_task_entropy_ema = self.task_entropy_ema
            else:
                self._prev_task_entropy_ema = self.task_entropy_ema
            self._train_call_count = 0

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef_task", float(np.mean(ent_coefs_task)) if ent_coefs_task else 0.0)
        self.logger.record("train/ent_coef_param", float(np.mean(ent_coefs_param)) if ent_coefs_param else 0.0)
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)) if actor_losses else 0.0)
        self.logger.record("train/bc_loss", float(np.mean(bc_losses)) if bc_losses else 0.0)
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)) if critic_losses else 0.0)
        if self.use_timing_profile:
            sample_time = float(np.mean(sample_times)) if sample_times else 0.0
            forward_time = float(np.mean(forward_times)) if forward_times else 0.0
            critic_time = float(np.mean(critic_times)) if critic_times else 0.0
            actor_time = float(np.mean(actor_times)) if actor_times else 0.0
            target_time = float(np.mean(target_times)) if target_times else 0.0
            entropy_time = float(np.mean(entropy_times)) if entropy_times else 0.0
            dex_time = float(np.mean(dex_times)) if dex_times else 0.0
            bc_time = float(np.mean(bc_times)) if bc_times else 0.0
            polyak_time = float(np.mean(polyak_times)) if polyak_times else 0.0

            self.logger.record("time/sample", sample_time)
            self.logger.record("time/forward", forward_time)
            self.logger.record("time/critic", critic_time)
            self.logger.record("time/actor", actor_time)
            self.logger.record("time/target", target_time)
            self.logger.record("time/entropy", entropy_time)
            self.logger.record("time/dex", dex_time)
            self.logger.record("time/bc", bc_time)
            self.logger.record("time/polyak", polyak_time)
            # Keep a stable copy for callbacks/scripts that read after logger flush.
            self._timing_snapshot = {
                "num_timesteps": int(self.num_timesteps),
                "sample_time": sample_time,
                "forward_time": forward_time,
                "critic_time": critic_time,
                "actor_time": actor_time,
                "target_time": target_time,
                "entropy_time": entropy_time,
                "dex_time": dex_time,
                "bc_time": bc_time,
                "polyak_time": polyak_time,
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                "bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
            }
        if len(ent_coef_task_losses) > 0:
            self.logger.record("train/ent_coef_task_loss", float(np.mean(ent_coef_task_losses)))
        if len(ent_coef_param_losses) > 0:
            self.logger.record("train/ent_coef_param_loss", float(np.mean(ent_coef_param_losses)))

        # debug
        self.logger.record("train/topk_dist_mean", float(np.mean(topk_dist_means)) if topk_dist_means else 0.0)
        self.logger.record("train/valid_demo_ratio", float(np.mean(valid_ratios)) if valid_ratios else 0.0)
        self.logger.record("train/max_topk_dist", float(np.mean(max_dists)) if max_dists else 0.0)
        self.logger.record("train/task_entropy", float(np.mean(task_entropys)) if task_entropys else 0.0)
        self.logger.record("train/critic_grad_norm", float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0)
        self.logger.record("train/actor_grad_norm", float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0)
        
        # 熵值EMA和变化率统计
        if self.task_entropy_ema is not None:
            self.logger.record("diagnostic/task_entropy_ema", self.task_entropy_ema)
            self.logger.record("diagnostic/task_entropy_change_rate", self._task_entropy_change_rate)
        
        # Q 值统计
        if target_q_means:
            self.logger.record("diagnostic/target_q_mean", float(np.mean(target_q_means)))
        if current_q_means:
            self.logger.record("diagnostic/current_q_mean", float(np.mean(current_q_means)))
        if target_q_re_means:
            self.logger.record("diagnostic/target_q_re_mean", float(np.mean(target_q_re_means)))
        if target_q_Tent_means:
            self.logger.record("diagnostic/target_q_Tent_mean", float(np.mean(target_q_Tent_means)))
        if target_q_dex_mins:
            self.logger.record("diagnostic/target_q_dex_mins", float(np.mean(target_q_dex_mins)))
        if target_q_Pent_means:
            self.logger.record("diagnostic/target_q_Pent_mean", float(np.mean(target_q_Pent_means)))

        # Reward 统计
        if reward_maxs:
            self.logger.record("diagnostic/reward_max", float(np.max(reward_maxs)))
        
        # 梯度爆炸诊断统计
        if gradient_spike_count > 0:
            self.logger.record("diagnostic/gradient_spike_count", gradient_spike_count)
        if max_td_error_list:
            self.logger.record("diagnostic/max_td_error", float(np.mean(max_td_error_list)))
        if mean_td_error_list:
            self.logger.record("diagnostic/mean_td_error", float(np.mean(mean_td_error_list)))
        if max_obs_val_list:
            self.logger.record("diagnostic/max_obs_val", float(np.mean(max_obs_val_list)))
        if max_desired_goal_val_list:
            self.logger.record("diagnostic/max_desired_goal_val", float(np.mean(max_desired_goal_val_list)))
        if max_achieved_goal_val_list:
            self.logger.record("diagnostic/max_achieved_goal_val", float(np.mean(max_achieved_goal_val_list)))
        
        # self.logger.record("train/demo_k", self.demo_k)
        # self.logger.record("train/demo_id_margin", self.demo_id_margin)