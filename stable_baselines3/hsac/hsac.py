import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.hsac.policies import HybridActor, HybridCritic, HybridSACPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.buffers import HybridReplayBuffer

SelfHSAC = TypeVar("SelfHSAC", bound="HSAC")


class HSAC(OffPolicyAlgorithm):
    """
    Hybrid Soft Actor-Critic (HSAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with Hybrid Actions.
    
    This extends SAC to handle hybrid (discrete + continuous) action spaces.
    Uses two entropy coefficients: alpha_task (discrete) and alpha_param (continuous).
    
    :param policy: The policy model to use (MlpPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: learning rate for adam optimizer
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every train_freq steps
    :param gradient_steps: How many gradient steps after each rollout
    :param action_noise: the action noise type (not typically used with SAC)
    :param replay_buffer_class: Replay buffer class (HybridReplayBuffer by default)
    :param replay_buffer_kwargs: Keyword arguments for replay buffer
    :param optimize_memory_usage: Enable memory efficient variant
    :param ent_coef_task: Entropy regularization coefficient for discrete actions
    :param ent_coef_param: Entropy regularization coefficient for continuous parameters
    :param target_update_interval: update the target network every N gradient steps
    :param target_entropy_task: target entropy for discrete actions ('auto' for automatic tuning)
    :param target_entropy_param: target entropy for continuous parameters ('auto' for automatic tuning)
    :param use_sde: Whether to use generalized State Dependent Exploration
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param use_sde_at_warmup: Whether to use gSDE during warm up phase
    :param stats_window_size: Window size for rollout logging
    :param tensorboard_log: the log location for tensorboard
    :param policy_kwargs: additional arguments for the policy
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...)
    :param _init_setup_model: Whether to build the network at creation
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: HybridSACPolicy
    actor: HybridActor
    critic: HybridCritic
    critic_target: HybridCritic

    def __init__(
        self,
        policy: Union[str, Type[HybridSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,        
        replay_buffer_class: Optional[Type[ReplayBuffer]] = HybridReplayBuffer, # Use HybridReplayBuffer by default
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef_task: Union[str, float] = "auto",
        ent_coef_param: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy_task: Union[str, float] = "auto",
        target_entropy_param: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(   
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Dict,),
            support_multi_env=True,
        )
        # Store original action space info before reorganization
        self.original_action_space = self.action_space

        # Reorganize action space to internal format
        self.action_space, d_key, c_key = self._reorganize_action_space(self.original_action_space)
        
        # Set policy kwargs (no longer passing n_discrete_actions and max_param_dim)
        self.policy_kwargs = policy_kwargs or {}
        self.policy_kwargs.update({
            "d_key": d_key,
            "c_key": c_key,
        })

        self.target_entropy_task = target_entropy_task
        self.target_entropy_param = target_entropy_param
        self.log_ent_coef_task = None
        self.log_ent_coef_param = None
        self.ent_coef_task = ent_coef_task
        self.ent_coef_param = ent_coef_param
        self.target_update_interval = target_update_interval
        self.ent_coef_task_optimizer: Optional[th.optim.Adam] = None
        self.ent_coef_param_optimizer: Optional[th.optim.Adam] = None

        self.d_key = d_key
        self.c_key = c_key
        # n_discrete_actions and max_param_dim will be set in _setup_model from policy

        if _init_setup_model:
            self._setup_model()

    def _reorganize_action_space(self, original_action_space: spaces.Dict) -> Tuple[spaces.Dict, str, str]:
        """
        Reorganize action space from Gym format (id, params0, params1, ...) 
        to HSAC internal format (discrete, continuous).
        
        :param original_action_space: Original Dict action space from Gym
        :return: (reorganized_space, d_key, c_key, n_discrete_actions, max_param_dim)
        """
        # Find discrete action key (usually 'id')
        d_key = None
        param_keys = []
        
        for key in original_action_space.spaces.keys():
            if isinstance(original_action_space.spaces[key], spaces.Discrete):
                d_key = key
            elif key.startswith('params'):
                param_keys.append(key)
        
        if d_key is None:
            raise ValueError("No discrete action found in action space")
        
        # Get number of discrete actions
        n_discrete_actions = original_action_space.spaces[d_key].n
        
        # Get max parameter dimension
        max_param_dim = 0
        for key in param_keys:
            param_space = original_action_space.spaces[key]
            if isinstance(param_space, spaces.Box):
                dim = int(np.prod(param_space.shape))
                max_param_dim = max(max_param_dim, dim)
        
        # Create reorganized action space
        c_key = "continuous"
        reorganized_space = spaces.Dict({
            "discrete": spaces.Discrete(n_discrete_actions),
            c_key: spaces.Box(low=-1.0, high=1.0, shape=(max_param_dim,), dtype=np.float32),
        })
        
        return reorganized_space, "discrete", c_key

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        
        # Initialize the policy's action restoration with the original action space
        if hasattr(self.policy, 'restore_action') and self.original_action_space is not None:
            self.policy.restore_action(action=None, original_action_space=self.original_action_space)
        
        # Get n_discrete_actions and max_param_dim from policy
        self.n_discrete_actions = self.policy.n_discrete_actions
        self.max_param_dim = self.policy.max_param_dim
        
        # Running mean and running var for batch norm
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        
        # Target entropy for discrete actions (task policy)
        if self.target_entropy_task == "auto":
            # For discrete: -log(1/n) = log(n)
            self.target_entropy_task = 0.98 * float(np.log(self.n_discrete_actions))
        else:
            self.target_entropy_task = float(self.target_entropy_task)
        
        # Target entropy for continuous parameters
        if self.target_entropy_param == "auto":
            # For continuous: -dim (standard SAC formula)
            self.target_entropy_param = float(-self.max_param_dim)
        else:
            self.target_entropy_param = float(self.target_entropy_param)
        
        # Entropy coefficient for task (discrete)
        if isinstance(self.ent_coef_task, str) and self.ent_coef_task.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef_task:
                init_value = float(self.ent_coef_task.split("_")[1])
            
            self.log_ent_coef_task = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_task_optimizer = th.optim.Adam([self.log_ent_coef_task], lr=self.lr_schedule(1))
        else:
            self.ent_coef_task_tensor = th.tensor(float(self.ent_coef_task), device=self.device)
        
        # Entropy coefficient for parameters (continuous)
        if isinstance(self.ent_coef_param, str) and self.ent_coef_param.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef_param:
                init_value = float(self.ent_coef_param.split("_")[1])
            
            self.log_ent_coef_param = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_param_optimizer = th.optim.Adam([self.log_ent_coef_param], lr=self.lr_schedule(1))
        else:
            self.ent_coef_param_tensor = th.tensor(float(self.ent_coef_param), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Implements the dual-entropy Hybrid SAC algorithm.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_task_optimizer is not None:
            optimizers.append(self.ent_coef_task_optimizer)
        if self.ent_coef_param_optimizer is not None:
            optimizers.append(self.ent_coef_param_optimizer)
        
        self._update_learning_rate(optimizers)

        ent_coef_task_losses, ent_coef_param_losses = [], []
        ent_coefs_task, ent_coefs_param = [], []
        actor_losses, critic_losses = [], []
        task_entropys = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Current actions and log probs from actor
            _, _, continuous_log_prob = self.actor.action_log_prob(replay_data.observations)

            # Get task distribution for current state
            logits = self.actor.get_task_dist_params(replay_data.observations)
            task_dist = th.distributions.Categorical(logits=logits)
            task_probs = task_dist.probs  # (batch_size, n_discrete_actions)
            # Log probabilities for all discrete actions
            task_log_probs = th.log_softmax(logits, dim=-1)  # (batch_size, n_discrete_actions)
            task_entropy = task_dist.entropy()  # (batch_size,)
            task_entropys.append(task_entropy.mean().item())

            # Get entropy coefficients
            if self.ent_coef_task_optimizer is not None and self.log_ent_coef_task is not None:
                ent_coef_task = th.exp(self.log_ent_coef_task.detach())
                ent_coef_task_loss = self.log_ent_coef_task * (task_entropy - self.target_entropy_task).detach().mean()
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
                # Compute target Q value using exact expectation over discrete actions
                # y = r + gamma*(1-d) * sum_a' [ pi_task(a'|s') * (min_i Q_targ(s',a',x') - alpha_task*log(pi_task(a'|s')) - alpha_param*log(pi_param(x'|s',a'))) ]
                
                # Get task distribution for next state
                next_logits = self.actor.get_task_dist_params(replay_data.next_observations)
                next_task_probs = th.softmax(next_logits, dim=-1)  # (batch_size, n_discrete_actions)
                next_task_log_probs = th.log_softmax(next_logits, dim=-1)  # (batch_size, n_discrete_actions)
                
                # Get all parameter distributions for next state
                # all_means, all_log_stds: (batch_size, n_discrete_actions, max_param_dim)
                all_next_means, all_next_log_stds = self.actor.get_all_param_dist_params(replay_data.next_observations)
                
                # Sample continuous actions for all discrete actions using reparameterization
                all_next_stds = th.exp(all_next_log_stds)
                noise = th.randn_like(all_next_means)
                all_next_continuous_pretanh = all_next_means + all_next_stds * noise
                all_next_continuous_actions = th.tanh(all_next_continuous_pretanh)  # (batch_size, n_discrete_actions, max_param_dim)
                
                # Compute log prob for continuous actions (tanh squashed gaussian)
                gaussian_log_prob = -0.5 * (
                    ((all_next_continuous_pretanh - all_next_means) / (all_next_stds + 1e-6)) ** 2 
                    + 2 * all_next_log_stds 
                    + np.log(2 * np.pi)
                )
                gaussian_log_prob = gaussian_log_prob.sum(dim=-1)  # (batch_size, n_discrete_actions)
                
                # Squashing correction
                squash_correction = th.log(1 - all_next_continuous_actions ** 2 + 1e-6).sum(dim=-1)
                all_next_continuous_log_probs = gaussian_log_prob - squash_correction  # (batch_size, n_discrete_actions)
                
                # Compute min Q-values across all critics for all discrete actions
                # all_next_q_values: (batch_size, n_discrete_actions)
                all_next_q_values = self.critic_target.forward_all_discrete(
                    replay_data.next_observations, all_next_continuous_actions
                )
                
                # Compute value for each discrete action: Q - alpha_task*log(pi_task) - alpha_param*log(pi_param)
                # (batch_size, n_discrete_actions)
                all_next_values = (
                    all_next_q_values 
                    - ent_coef_task * next_task_log_probs 
                    - ent_coef_param * all_next_continuous_log_probs
                )
                
                # Compute expected value by summing over discrete actions weighted by task policy
                # next_v = sum_a' [ pi_task(a'|s') * V(s', a') ]
                next_v = (next_task_probs * all_next_values).sum(dim=1, keepdim=True)  # (batch_size, 1)
                
                # Compute target Q values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v

            # Get current Q-values estimates
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss - VECTORIZED VERSION
            # For hybrid SAC: we marginalize over discrete actions
            # actor_loss = E_a[pi(a|s) * (alpha_task*log(pi(a|s)) + alpha_param*log(pi(x|s,a)) - Q(s,a,x))]
            
            # Get all parameter distributions at once
            # all_means, all_log_stds: (batch_size, n_discrete_actions, max_param_dim)
            all_means, all_log_stds = self.actor.get_all_param_dist_params(replay_data.observations)
            
            # Sample continuous actions for all discrete actions using reparameterization trick
            all_stds = th.exp(all_log_stds)
            noise = th.randn_like(all_means)
            all_continuous_pretanh = all_means + all_stds * noise
            all_continuous_actions = th.tanh(all_continuous_pretanh)
            
            # Compute log prob for continuous actions (tanh squashed gaussian)
            # log_prob = N(pretanh|mean, std).log_prob - log(1 - tanh(pretanh)^2)
            gaussian_log_prob = -0.5 * (
                ((all_continuous_pretanh - all_means) / (all_stds + 1e-6)) ** 2 
                + 2 * all_log_stds 
                + np.log(2 * np.pi)
            )
            gaussian_log_prob = gaussian_log_prob.sum(dim=-1)  # (batch_size, n_discrete_actions)
            
            # Squashing correction
            squash_correction = th.log(1 - all_continuous_actions ** 2 + 1e-6).sum(dim=-1)
            all_continuous_log_probs = gaussian_log_prob - squash_correction  # (batch_size, n_discrete_actions)
            
            # Compute Q-values for all discrete actions at once
            # all_q_values: (batch_size, n_discrete_actions, 1)
            # q_1    version:
            # all_q_values = self.critic.q1_forward_all_discrete(replay_data.observations, all_continuous_actions)
            # all_q_values = all_q_values.squeeze(-1)  # (batch_size, n_discrete_actions)
            
            # q_min  version:
            all_q_values = self.critic.forward_all_discrete(replay_data.observations, all_continuous_actions)

            # Compute weighted actor loss
            # actor_loss = sum_a [ pi(a|s) * (alpha_task * log(pi(a|s)) + alpha_param * log(pi(x|s,a)) - Q(s,a,x)) ]
            weighted_loss = task_probs * (
                ent_coef_task * task_log_probs +
                ent_coef_param * all_continuous_log_probs -
                all_q_values
            )  # (batch_size, n_discrete_actions)
            actor_loss = weighted_loss.sum(dim=1).mean()  # Sum over discrete actions, mean over batch
            
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef_task", np.mean(ent_coefs_task))
        self.logger.record("train/ent_coef_param", np.mean(ent_coefs_param))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/task_entropy", float(np.mean(task_entropys)) if task_entropys else 0.0)
        if len(ent_coef_task_losses) > 0:
            self.logger.record("train/ent_coef_task_loss", np.mean(ent_coef_task_losses))
        if len(ent_coef_param_losses) > 0:
            self.logger.record("train/ent_coef_param_loss", np.mean(ent_coef_param_losses))

    def learn(
        self: SelfHSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "HSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = []
        
        if self.ent_coef_task_optimizer is not None:
            state_dicts.append("ent_coef_task_optimizer")
            saved_pytorch_variables.append("log_ent_coef_task")
        else:
            saved_pytorch_variables.append("ent_coef_task_tensor")
        
        if self.ent_coef_param_optimizer is not None:
            state_dicts.append("ent_coef_param_optimizer")
            saved_pytorch_variables.append("log_ent_coef_param")
        else:
            saved_pytorch_variables.append("ent_coef_param_tensor")
        
        return state_dicts, saved_pytorch_variables
