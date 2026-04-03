from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class HybridActor(BasePolicy):
    """
    Hybrid Actor network (policy) for Hybrid SAC.
    
    Hierarchical structure:
    - Task Policy: outputs discrete action (categorical distribution)
    - Parameter Policy: outputs continuous parameters conditioned on discrete action
    
    :param observation_space: Observation space
    :param action_space: Action space (must be spaces.Dict)
    :param net_arch: Network architecture for both task and parameter networks
    :param features_extractor: Network to extract features
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not
    :param d_key: Key for discrete action in action space Dict
    :param c_key: Key prefix for continuous parameters in action space Dict
    :param n_discrete_actions: Number of discrete actions
    :param max_param_dim: Maximum dimension of continuous parameters
    :param param_mask: Mask for valid parameter dimensions (n_discrete_actions, max_param_dim)
    """

    action_space: spaces.Dict

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Dict,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        d_key: str = "discrete",
        c_key: str = "continuous",
        discrete_epsilon: float = 0.0,
        param_mask: Optional[np.ndarray] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.d_key = d_key
        self.c_key = c_key
        self.discrete_epsilon = discrete_epsilon  # Epsilon for discrete action exploration
        
        # Get n_discrete_actions from action space
        assert isinstance(action_space, spaces.Dict), "Action space must be Dict"
        assert d_key in action_space.spaces, f"Key {d_key} not found in action space"
        assert isinstance(action_space.spaces[d_key], spaces.Discrete), "Discrete action must be Discrete space"
        self.n_discrete_actions = action_space.spaces[d_key].n
        
        # Get continuous parameter dimension
        assert c_key in action_space.spaces, f"Key {c_key} not found in action space"
        assert isinstance(action_space.spaces[c_key], spaces.Box), "Continuous action must be Box space"
        self.max_param_dim = int(np.prod(action_space.spaces[c_key].shape))

        # Initialize param_mask (will be moved to correct device later)
        # param_mask: (n_discrete_actions, max_param_dim) - 1.0 for valid dims, 0.0 otherwise
        if param_mask is not None:
            self.register_buffer(
                "param_mask", 
                th.as_tensor(param_mask, dtype=th.float32)
            )
        else:
            # Default: all dimensions are valid for all actions
            self.register_buffer(
                "param_mask",
                th.ones(self.n_discrete_actions, self.max_param_dim, dtype=th.float32)
            )

        # Action distribution for continuous parameters (similar to SAC)
        self.param_action_dist = SquashedDiagGaussianDistribution(self.max_param_dim, epsilon=1e-6)

        # Task policy network (discrete action)
        task_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.task_latent = nn.Sequential(*task_net)
        last_layer_dim_task = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.task_logits = nn.Linear(last_layer_dim_task, self.n_discrete_actions)

        # Parameter policy network: one sub-network for each discrete action
        # Each sub-network takes features as input and outputs parameters for that action
        self.param_networks = nn.ModuleList()
        for _ in range(self.n_discrete_actions):
            param_net = create_mlp(features_dim, -1, net_arch, activation_fn)
            param_latent = nn.Sequential(*param_net)
            last_layer_dim_param = net_arch[-1] if len(net_arch) > 0 else features_dim
            
            # Each sub-network outputs mean and log_std for max_param_dim
            mu = nn.Linear(last_layer_dim_param, self.max_param_dim)
            log_std = nn.Linear(last_layer_dim_param, self.max_param_dim)
            
            self.param_networks.append(nn.ModuleDict({
                'latent': param_latent,
                'mu': mu,
                'log_std': log_std,
            }))

    def get_task_dist_params(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the parameters for task (discrete action) distribution.
        Similar to SAC's get_action_dist_params.
        
        :param obs: Observation
        :return: Logits for categorical distribution
        """
        features = self.extract_features(obs, self.features_extractor)
        task_latent = self.task_latent(features)
        logits = self.task_logits(task_latent)
        return logits

    def get_param_dist_params(
        self, obs: PyTorchObs, discrete_action: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for parameter (continuous) distribution conditioned on discrete action.
        Similar to SAC's get_action_dist_params.
        Uses pre-computed outputs from all sub-networks and indexes by discrete action.
        
        :param obs: Observation
        :param discrete_action: Discrete action (batch_size,)
        :return: Mean, log_std, and optional keyword arguments
        """
        features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]
        
        # Pre-compute outputs from all sub-networks
        # This avoids dynamic masking and allows efficient batch indexing
        all_means = []
        all_log_stds = []
        
        for action_idx in range(self.n_discrete_actions):
            param_net = self.param_networks[action_idx]
            latent = param_net['latent'](features)
            all_means.append(param_net['mu'](latent))
            all_log_stds.append(param_net['log_std'](latent))
        
        # Stack to (batch_size, n_discrete_actions, max_param_dim)
        all_means = th.stack(all_means, dim=1)
        all_log_stds = th.stack(all_log_stds, dim=1)
        
        # Index by discrete action to get the corresponding mean and log_std
        # discrete_action: (batch_size,) -> expand to (batch_size, 1, max_param_dim) for gather
        batch_indices = th.arange(batch_size, device=features.device)
        mean = all_means[batch_indices, discrete_action.long()]  # (batch_size, max_param_dim)
        log_std = all_log_stds[batch_indices, discrete_action.long()]  # (batch_size, max_param_dim)
        
        # Clamp log_std (similar to SAC)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std, {}

    def get_all_param_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the parameters for ALL discrete actions' parameter distributions.
        This is used for vectorized actor loss computation.
        
        :param obs: Observation (batch_size, obs_dim)
        :return: Mean and log_std for all discrete actions
                 mean: (batch_size, n_discrete_actions, max_param_dim)
                 log_std: (batch_size, n_discrete_actions, max_param_dim)
        """
        features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]
        
        # Pre-allocate tensors for all discrete actions
        all_means = th.zeros(batch_size, self.n_discrete_actions, self.max_param_dim, device=features.device)
        all_log_stds = th.zeros(batch_size, self.n_discrete_actions, self.max_param_dim, device=features.device)
        
        # Compute for each discrete action (no masking needed, all actions computed)
        for action_idx in range(self.n_discrete_actions):
            param_net = self.param_networks[action_idx]
            latent = param_net['latent'](features)
            all_means[:, action_idx, :] = param_net['mu'](latent)
            all_log_stds[:, action_idx, :] = param_net['log_std'](latent)
        
        # Clamp log_std
        all_log_stds = th.clamp(all_log_stds, LOG_STD_MIN, LOG_STD_MAX)
        
        return all_means, all_log_stds

    def evaluate_all_actions(
        self, obs: PyTorchObs
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Unified evaluation method for all discrete actions with param_mask applied.
        This integrates reparameterization and probability computation for both
        Target Q computation and Actor Loss computation.
        
        :param obs: Observation (batch_size, obs_dim)
        :return: Tuple of:
            - task_probs: (batch_size, n_discrete_actions)
            - task_log_probs: (batch_size, n_discrete_actions)
            - task_entropy: (batch_size,)
            - all_continuous_actions: (batch_size, n_discrete_actions, max_param_dim) with mask applied
            - all_continuous_log_probs: (batch_size, n_discrete_actions) with log-prob mask applied
        """
        # Get task distribution
        logits = self.get_task_dist_params(obs)
        task_dist = th.distributions.Categorical(logits=logits)
        task_probs = task_dist.probs  # (batch_size, n_discrete_actions)
        task_log_probs = th.log_softmax(logits, dim=-1)  # (batch_size, n_discrete_actions)
        task_entropy = task_dist.entropy()  # (batch_size,)
        
        # Get all parameter distributions
        all_means, all_log_stds = self.get_all_param_dist_params(obs)
        # all_means, all_log_stds: (batch_size, n_discrete_actions, max_param_dim)
        
        # Reparameterization: sample continuous actions for all discrete actions
        all_stds = th.exp(all_log_stds)
        noise = th.randn_like(all_means)
        all_continuous_pretanh = all_means + all_stds * noise
        
        # Apply action mask: mask out invalid dimensions
        # param_mask: (n_discrete_actions, max_param_dim) -> expand to (1, n_discrete_actions, max_param_dim)
        all_continuous_actions = th.tanh(all_continuous_pretanh) * self.param_mask.unsqueeze(0)
        
        # Compute log prob for continuous actions (tanh squashed gaussian)
        # gaussian_log_prob: (batch_size, n_discrete_actions, max_param_dim)
        gaussian_log_prob = -0.5 * (
            ((all_continuous_pretanh - all_means) / (all_stds + 1e-6)) ** 2 
            + 2 * all_log_stds 
            + np.log(2 * np.pi)
        )
        
        # Squashing correction: (batch_size, n_discrete_actions, max_param_dim)
        squash_correction = th.log(1 - all_continuous_actions ** 2 + 1e-6)
        
        # Apply log-prob mask before summing
        # Only sum over valid dimensions for each discrete action
        all_continuous_log_probs = (
            (gaussian_log_prob - squash_correction) * self.param_mask.unsqueeze(0)
        ).sum(dim=-1)  # (batch_size, n_discrete_actions)
        
        return task_probs, task_log_probs, task_entropy, all_continuous_actions, all_continuous_log_probs

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[Dict[str, th.Tensor], th.Tensor, th.Tensor]:
        """
        Forward pass: sample both discrete and continuous actions.
        Similar to SAC's forward method.
        
        :param obs: Observation
        :param deterministic: Whether to use deterministic actions
        :return: Dictionary of actions, discrete log prob, continuous log prob
        """
        # Get task distribution parameters and sample discrete action
        logits = self.get_task_dist_params(obs)
        task_dist = th.distributions.Categorical(logits=logits)
        
        if deterministic:
            discrete_action = th.argmax(task_dist.probs, dim=1)
        else:
            # Epsilon-greedy exploration for discrete actions
            batch_size = obs.shape[0] if isinstance(obs, th.Tensor) else obs['observation'].shape[0]
            
            # Sample from uniform distribution with probability epsilon
            if self.discrete_epsilon > 0:
                # Generate random mask for exploration
                explore_mask = th.rand(batch_size, device=logits.device) < self.discrete_epsilon
                
                # Sample from policy
                policy_action = task_dist.sample()
                
                # Sample uniformly from action space
                uniform_action = th.randint(0, self.n_discrete_actions, (batch_size,), device=logits.device)
                
                # Mix exploration and exploitation
                discrete_action = th.where(explore_mask, uniform_action, policy_action)
            else:
                discrete_action = task_dist.sample()
        
        discrete_log_prob = task_dist.log_prob(discrete_action)
        
        # Get parameter distribution parameters
        mean_actions, log_std, _ = self.get_param_dist_params(obs, discrete_action)
        
        # Get current mask for selected discrete actions: (batch_size, max_param_dim)
        current_mask = self.param_mask[discrete_action.long()]
        
        # Sample continuous action with reparameterization
        std = th.exp(log_std)
        if deterministic:
            continuous_pretanh = mean_actions
        else:
            noise = th.randn_like(mean_actions)
            continuous_pretanh = mean_actions + std * noise
        
        # Apply action mask
        continuous_action = th.tanh(continuous_pretanh) * current_mask
        
        # Compute log prob with mask applied
        gaussian_log_prob = -0.5 * (
            ((continuous_pretanh - mean_actions) / (std + 1e-6)) ** 2 
            + 2 * log_std 
            + np.log(2 * np.pi)
        )
        squash_correction = th.log(1 - continuous_action ** 2 + 1e-6)
        continuous_log_prob = ((gaussian_log_prob - squash_correction) * current_mask).sum(dim=-1)
        
        actions = {
            self.d_key: discrete_action,
            self.c_key: continuous_action,
        }
        
        return actions, discrete_log_prob, continuous_log_prob

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[Dict[str, th.Tensor], th.Tensor, th.Tensor]:
        """
        Sample actions and compute log probabilities (for training).
        Similar to SAC's action_log_prob method.
        
        :param obs: Observation
        :return: Actions, discrete log prob, continuous log prob
        """
        # Get task distribution parameters and sample discrete action
        logits = self.get_task_dist_params(obs)
        task_dist = th.distributions.Categorical(logits=logits)
        discrete_action = task_dist.sample()
        discrete_log_prob = task_dist.log_prob(discrete_action)
        
        # Get parameter distribution parameters
        mean_actions, log_std, _ = self.get_param_dist_params(obs, discrete_action)
        
        # Get current mask for selected discrete actions: (batch_size, max_param_dim)
        current_mask = self.param_mask[discrete_action.long()]
        
        # Sample continuous action with reparameterization
        std = th.exp(log_std)
        noise = th.randn_like(mean_actions)
        continuous_pretanh = mean_actions + std * noise
        
        # Apply action mask
        continuous_action = th.tanh(continuous_pretanh) * current_mask
        
        # Compute log prob with mask applied
        gaussian_log_prob = -0.5 * (
            ((continuous_pretanh - mean_actions) / (std + 1e-6)) ** 2 
            + 2 * log_std 
            + np.log(2 * np.pi)
        )
        squash_correction = th.log(1 - continuous_action ** 2 + 1e-6)
        continuous_log_prob = ((gaussian_log_prob - squash_correction) * current_mask).sum(dim=-1)
        
        actions = {
            self.d_key: discrete_action,
            self.c_key: continuous_action,
        }
        
        return actions, discrete_log_prob, continuous_log_prob

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> Dict[str, th.Tensor]:
        """
        Predict action (used by BasePolicy.predict).
        
        :param observation: Observation
        :param deterministic: Whether to use deterministic actions
        :return: Dictionary of actions
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions


class HybridCritic(BaseModel):
    """
    Hybrid Critic network (Q-function) for Hybrid SAC.
    
    Takes (observation, discrete_action, continuous_action) as input.
    
    :param observation_space: Observation space
    :param action_space: Action space (must be spaces.Dict)
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not
    :param n_critics: Number of critic networks (typically 2 for SAC)
    :param d_key: Key for discrete action in action space Dict
    :param c_key: Key for continuous parameters in action space Dict
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Dict,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        d_key: str = "discrete",
        c_key: str = "continuous",
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.d_key = d_key
        self.c_key = c_key
        
        # Get parameters from action space
        assert isinstance(action_space, spaces.Dict), "Action space must be Dict"
        self.n_discrete_actions = action_space.spaces[d_key].n
        self.max_param_dim = int(np.prod(action_space.spaces[c_key].shape))

        # Input: features + one-hot discrete action + continuous action
        q_input_dim = features_dim + self.n_discrete_actions + self.max_param_dim

        # Create multiple Q-networks
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(q_input_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: PyTorchObs, actions: Dict[str, th.Tensor]) -> Tuple[th.Tensor, ...]:
        """
        Forward pass through all Q-networks.
        
        :param obs: Observation
        :param actions: Dictionary with discrete and continuous actions
        :return: Tuple of Q-values from each critic
        """
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        
        # Get discrete and continuous actions
        discrete_action = actions[self.d_key]
        continuous_action = actions[self.c_key]
        
        # One-hot encode discrete action
        discrete_one_hot = F.one_hot(discrete_action.long().flatten(), num_classes=self.n_discrete_actions).float()
        
        # Concatenate features, one-hot discrete action, and continuous action
        q_input = th.cat([features, discrete_one_hot, continuous_action], dim=1)
        
        # Compute Q-values from each critic
        return tuple(q_net(q_input) for q_net in self.q_networks)

    def q1_forward(self, obs: PyTorchObs, actions: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        
        :param obs: Observation
        :param actions: Dictionary with discrete and continuous actions
        :return: Q-value from first critic
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        
        discrete_action = actions[self.d_key]
        continuous_action = actions[self.c_key]
        
        discrete_one_hot = F.one_hot(discrete_action.long().flatten(), num_classes=self.n_discrete_actions).float()
        q_input = th.cat([features, discrete_one_hot, continuous_action], dim=1)
        
        return self.q_networks[0](q_input)

    def q1_forward_all_discrete(
        self, obs: PyTorchObs, all_continuous_actions: th.Tensor
    ) -> th.Tensor:
        """
        Compute Q-values for all discrete actions at once using the first Q-network.
        This is used for vectorized actor loss computation.
        
        :param obs: Observation (batch_size, obs_dim)
        :param all_continuous_actions: Continuous actions for all discrete actions
                                       (batch_size, n_discrete_actions, max_param_dim)
        :return: Q-values for all discrete actions (batch_size, n_discrete_actions, 1)
        """
        # Extract features (no gradient through features extractor for actor update)
        features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]
        
        # Expand features for all discrete actions: (batch_size, n_discrete_actions, features_dim)
        features_expanded = features.unsqueeze(1).expand(-1, self.n_discrete_actions, -1)
        
        # Create one-hot for all discrete actions: (n_discrete_actions, n_discrete_actions)
        all_one_hots = th.eye(self.n_discrete_actions, device=features.device)
        # Expand to batch: (batch_size, n_discrete_actions, n_discrete_actions)
        all_one_hots = all_one_hots.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: (batch_size, n_discrete_actions, features_dim + n_discrete_actions + max_param_dim)
        q_input = th.cat([features_expanded, all_one_hots, all_continuous_actions], dim=-1)
        
        # Reshape for batch processing: (batch_size * n_discrete_actions, input_dim)
        q_input_flat = q_input.reshape(-1, q_input.shape[-1])
        
        # Compute Q-values using first Q-network
        q_flat = self.q_networks[0](q_input_flat)  # (batch_size * n_discrete_actions, 1)
        q_values = q_flat.reshape(batch_size, self.n_discrete_actions, 1)
        
        return q_values

    def forward_all_discrete(
        self, obs: PyTorchObs, all_continuous_actions: th.Tensor
    ) -> th.Tensor:
        """
        Compute Q-values for all discrete actions using ALL Q-networks,
        and return the minimum Q-value across all critics.
        This is used for computing accurate expected target Q-values.
        
        :param obs: Observation (batch_size, obs_dim)
        :param all_continuous_actions: Continuous actions for all discrete actions
                                       (batch_size, n_discrete_actions, max_param_dim)
        :return: Minimum Q-values across all critics for all discrete actions 
                 (batch_size, n_discrete_actions)
        """
        # Extract features
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]
        
        # Expand features for all discrete actions: (batch_size, n_discrete_actions, features_dim)
        features_expanded = features.unsqueeze(1).expand(-1, self.n_discrete_actions, -1)
        
        # Create one-hot for all discrete actions: (n_discrete_actions, n_discrete_actions)
        all_one_hots = th.eye(self.n_discrete_actions, device=features.device)
        # Expand to batch: (batch_size, n_discrete_actions, n_discrete_actions)
        all_one_hots = all_one_hots.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: (batch_size, n_discrete_actions, features_dim + n_discrete_actions + max_param_dim)
        q_input = th.cat([features_expanded, all_one_hots, all_continuous_actions], dim=-1)
        
        # Reshape for batch processing: (batch_size * n_discrete_actions, input_dim)
        q_input_flat = q_input.reshape(-1, q_input.shape[-1])
        
        # Compute Q-values from all Q-networks and take minimum
        all_q_values = []
        for q_net in self.q_networks:
            q_flat = q_net(q_input_flat)  # (batch_size * n_discrete_actions, 1)
            q_values = q_flat.reshape(batch_size, self.n_discrete_actions)  # (batch_size, n_discrete_actions)
            all_q_values.append(q_values)
        
        # Stack and take minimum: (n_critics, batch_size, n_discrete_actions) -> (batch_size, n_discrete_actions)
        stacked_q = th.stack(all_q_values, dim=0)
        min_q_values, _ = th.min(stacked_q, dim=0)
        
        return min_q_values


class HybridSACPolicy(BasePolicy):
    """
    Policy class for Hybrid SAC algorithm.
    
    :param observation_space: Observation space
    :param action_space: Action space (must be spaces.Dict)
    :param lr_schedule: Learning rate schedule
    :param net_arch: Network architecture
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor class
    :param features_extractor_kwargs: Features extractor kwargs
    :param normalize_images: Whether to normalize images
    :param optimizer_class: Optimizer class
    :param optimizer_kwargs: Optimizer kwargs
    :param n_critics: Number of critic networks
    :param d_key: Key for discrete action
    :param c_key: Key for continuous parameters
    :param n_discrete_actions: Number of discrete actions
    :param max_param_dim: Maximum dimension of continuous parameters
    :param param_mask: Mask for valid parameter dimensions (n_discrete_actions, max_param_dim)
    """

    actor: HybridActor
    critic: HybridCritic
    critic_target: HybridCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Dict,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        d_key: str = "discrete",
        c_key: str = "continuous",
        share_features_extractor: bool = False,
        discrete_epsilon: float = 0.0,
        param_mask: Optional[np.ndarray] = None,
        freeze_state_and_post_mlp: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        # Note: use_sde is not used in HSAC, but accepted for compatibility
        self.use_sde = use_sde
        self.d_key = d_key
        self.c_key = c_key
        self.discrete_epsilon = discrete_epsilon
        self.param_mask = param_mask
        self.freeze_state_and_post_mlp = freeze_state_and_post_mlp
        
        # Get parameters from action space
        assert isinstance(action_space, spaces.Dict), "Action space must be Dict"
        self.n_discrete_actions = action_space.spaces[d_key].n
        self.max_param_dim = int(np.prod(action_space.spaces[c_key].shape))
        
        # For action restoration
        self._type_key = None
        self._parameter_key = None
        self._parameter_dims = []
        self._parameter_lows = []
        self._parameter_highs = []

        # Default network architecture
        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        self.net_arch = net_arch
        activation_fn_by_name = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU
        }
        self.activation_fn = activation_fn_by_name[activation_fn] if isinstance(activation_fn, str) else activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update({
            "discrete_epsilon": self.discrete_epsilon,
            "param_mask": self.param_mask,
        })
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    @staticmethod
    def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _set_extractor_condition_only_trainable(self, extractor: Optional[nn.Module]) -> None:
        if extractor is None:
            return

        # Freeze the state stream, keep conditioning path trainable.
        if hasattr(extractor, "state_mlp"):
            self._set_module_requires_grad(getattr(extractor, "state_mlp"), False)
        if hasattr(extractor, "cond_mlp"):
            self._set_module_requires_grad(getattr(extractor, "cond_mlp"), True)
        if hasattr(extractor, "gamma"):
            self._set_module_requires_grad(getattr(extractor, "gamma"), True)
        if hasattr(extractor, "beta"):
            self._set_module_requires_grad(getattr(extractor, "beta"), True)

    def _apply_condition_only_freeze(self) -> None:
        if not self.freeze_state_and_post_mlp:
            return

        # Actor: freeze latent heads after extractor; keep conditioning extractor trainable.
        self._set_extractor_condition_only_trainable(getattr(self.actor, "features_extractor", None))
        self._set_module_requires_grad(getattr(self.actor, "task_latent", None), False)
        self._set_module_requires_grad(getattr(self.actor, "task_logits", None), False)
        for module_dict in getattr(self.actor, "param_networks", []):
            self._set_module_requires_grad(module_dict, False)

        # Critic: freeze post-feature Q MLP(s); keep conditioning extractor trainable.
        self._set_extractor_condition_only_trainable(getattr(self.critic, "features_extractor", None))
        for q_net in getattr(self.critic, "q_networks", []):
            self._set_module_requires_grad(q_net, False)

        self._set_extractor_condition_only_trainable(getattr(self.critic_target, "features_extractor", None))
        for q_net in getattr(self.critic_target, "q_networks", []):
            self._set_module_requires_grad(q_net, False)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build networks."""
        # Create actor
        self.actor = self.make_actor()

        # Create critics (Q-networks)
        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self._apply_condition_only_freeze()

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        
        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> HybridActor:
        """Create actor network."""
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor=features_extractor
        )
        return HybridActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> HybridCritic:
        """Create critic network."""
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor=features_extractor
        )
        return HybridCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Dict[str, th.Tensor]:
        """Forward pass."""
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> Dict[str, th.Tensor]:
        """Predict action."""
        return self.actor._predict(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def _predict_for_buffer(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """
        Get the policy action in internal format (for storing in replay buffer).
        Returns action in internal Dict format without restore_action transformation.
        
        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action in internal format and None (for state compatibility)
        """
        # Switch to eval mode
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)

        # Convert tensor to numpy, and reshape to the original action shape
        for key, act in actions.items():
            actions[key] = act.cpu().numpy().reshape((-1, *self.action_space.spaces[key].shape))

        # Handle continuous actions with squashing
        for key, act in actions.items():
            if isinstance(self.action_space.spaces[key], spaces.Box):
                if self.squash_output:
                    # Rescale to proper domain when using squashing
                    actions[key] = self.unscale_action(act, self.action_space.spaces[key])
                else:
                    # Clip actions to avoid out of bound error
                    actions[key] = np.clip(act, self.action_space.spaces[key].low, self.action_space.spaces[key].high)

        # Remove batch dimension if needed
        if not vectorized_env:
            for key, act in actions.items():
                actions[key] = act.squeeze(axis=0)

        return actions, None

    def restore_action(
        self,
        action: Optional[Union[Dict, List[Dict]]] = None,
        original_action_space: Optional[spaces.Dict] = None,
    ):
        """
        Restore the internal action format to the original gym environment format.
        Internal format: Dict(discrete=Discrete(n), continuous=Box(max_dim))
        Original format: Dict(id=Discrete(n), params0=Box(...), params1=Box(...), ...)
        
        All parameter fields are returned, with unused parameters filled with -1.
        
        :param action: The action to restore (None for initialization)
        :param original_action_space: The original action space (for initialization)
        :return: Restored action in original format
        """
        import re
        
        # Initialize if needed
        if original_action_space is not None and (self._type_key is None or self._parameter_key is None):
            self._parameter_dims = []
            self._parameter_lows = []
            self._parameter_highs = []
            for key, space in original_action_space.spaces.items():
                if isinstance(space, spaces.Discrete):
                    self._type_key = key
                else:
                    # Extract parameter key prefix (e.g., 'params' from 'params0')
                    if self._parameter_key is None:
                        self._parameter_key = re.split(r'(\d+)', key)[0]
                    # Store dimension and bounds of each parameter
                    param_dim = int(np.prod(space.shape))
                    self._parameter_dims.append(param_dim)
                    if isinstance(space, spaces.Box):
                        self._parameter_lows.append(space.low.flatten())
                        self._parameter_highs.append(space.high.flatten())
                    else:
                        # Fallback for non-Box spaces
                        self._parameter_lows.append(np.full(param_dim, -1.0))
                        self._parameter_highs.append(np.full(param_dim, 1.0))
            
            print(f"TYPE_KEY: {self._type_key}, PARAMETER_KEY: {self._parameter_key}, PARAMETER_DIMS: {self._parameter_dims}")
        
        if action is None:
            return None
        
        # Restore single action
        if isinstance(action, dict):
            discrete_action = int(action[self.d_key])
            continuous_params = action[self.c_key]
            
            # Build the restored action with all parameter fields
            restored = {self._type_key: discrete_action}
            
            # Add all parameter fields (fill unused ones with -1)
            for param_idx in range(len(self._parameter_dims)):
                param_dim = self._parameter_dims[param_idx]
                if param_idx == discrete_action:
                    # Use actual parameters for the selected action and denormalize
                    normalized_params = continuous_params[:param_dim]
                    # Denormalize from [-1, 1] to original range
                    low = self._parameter_lows[param_idx]
                    high = self._parameter_highs[param_idx]
                    params = (normalized_params + 1) / 2 * (high - low) + low
                else:
                    # Fill unused parameters with original space's low values
                    low = self._parameter_lows[param_idx]
                    params = low.copy()  # Use low values instead of -1
                
                restored[f"{self._parameter_key}{param_idx}"] = params
            
            return restored
        
        # Restore batch of actions
        else:
            restored_actions = []
            for a in action:
                discrete_action = int(a[self.d_key])
                continuous_params = a[self.c_key]
                
                # Build the restored action with all parameter fields
                restored = {self._type_key: discrete_action}
                
                # Add all parameter fields (fill unused ones with -1)
                for param_idx in range(len(self._parameter_dims)):
                    param_dim = self._parameter_dims[param_idx]
                    if param_idx == discrete_action:
                        # Use actual parameters for the selected action and denormalize
                        normalized_params = continuous_params[:param_dim]
                        # Denormalize from [-1, 1] to original range
                        low = self._parameter_lows[param_idx]
                        high = self._parameter_highs[param_idx]
                        params = (normalized_params + 1) / 2 * (high - low) + low
                    else:
                        # Fill unused parameters with original space's low values
                        low = self._parameter_lows[param_idx]
                        params = low.copy()  # Use low values instead of -1
                    
                    restored[f"{self._parameter_key}{param_idx}"] = params
                
                restored_actions.append(restored)
            
            return restored_actions


class FiLMHITLExtractor(BaseFeaturesExtractor):
    """FiLM feature extractor for HITL_PegTransfer-style dict observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 128,
        cond_dim: int = 7,
        obs_key: str = "observation",
        achieved_key: str = "achieved_goal",
        desired_key: str = "desired_goal",
    ):
        assert isinstance(observation_space, spaces.Dict), "FiLMHITLExtractor requires Dict observation space"
        assert obs_key in observation_space.spaces, f"Missing key in observation space: {obs_key}"
        assert achieved_key in observation_space.spaces, f"Missing key in observation space: {achieved_key}"
        assert desired_key in observation_space.spaces, f"Missing key in observation space: {desired_key}"

        self.obs_key = obs_key
        self.achieved_key = achieved_key
        self.desired_key = desired_key
        self.cond_dim = cond_dim

        obs_dim = int(np.prod(observation_space.spaces[obs_key].shape))
        achieved_dim = int(np.prod(observation_space.spaces[achieved_key].shape))
        desired_dim = int(np.prod(observation_space.spaces[desired_key].shape))
        assert obs_dim > cond_dim, f"obs_dim ({obs_dim}) must be > cond_dim ({cond_dim})"

        state_obs_dim = obs_dim - cond_dim
        self.state_in_dim = state_obs_dim + achieved_dim + desired_dim

        super().__init__(observation_space, features_dim=hidden_dim)

        # State stream
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Conditioning stream
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # FiLM heads for a single modulation stage.
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

        # Near-identity FiLM init improves early training stability.
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

        # Runtime stats for external logging (updated every forward call).
        self.last_gamma_mean = 1.0
        self.last_beta_mean = 0.0

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        obs_vec = observations[self.obs_key]
        cond = obs_vec[:, -self.cond_dim :]
        state_main = obs_vec[:, :-self.cond_dim]

        x_state = th.cat(
            [
                state_main,
                observations[self.achieved_key],
                observations[self.desired_key],
            ],
            dim=1,
        )

        c = self.cond_mlp(cond)

        h = self.state_mlp(x_state)
        gamma = self.gamma(c)
        beta = self.beta(c)
        modulated = gamma * h + beta
        h = F.relu(modulated)

        with th.no_grad():
            self.last_gamma_mean = float(gamma.mean().item())
            self.last_beta_mean = float(beta.mean().item())

        return h

    def get_film_stats(self) -> Dict[str, float]:
        return {
            "film/gamma_mean": self.last_gamma_mean,
            "film/beta_mean": self.last_beta_mean,
        }


class MultiInputPolicy(HybridSACPolicy):
    """
    Policy class (with both actor and critic) for HybridSAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    :param param_mask: Mask for valid parameter dimensions (n_discrete_actions, max_param_dim)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Dict,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        d_key: str = "discrete",
        c_key: str = "continuous",
        share_features_extractor: bool = False,
        discrete_epsilon: float = 0.0,
        param_mask: Optional[np.ndarray] = None,
        freeze_state_and_post_mlp: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            d_key,
            c_key,
            share_features_extractor,
            discrete_epsilon,
            param_mask,
            freeze_state_and_post_mlp,
        )


class FiLMMultiInputPolicy(HybridSACPolicy):
    """HybridSAC policy that uses FiLMHITLExtractor by default."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Dict,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FiLMHITLExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        d_key: str = "discrete",
        c_key: str = "continuous",
        share_features_extractor: bool = False,
        discrete_epsilon: float = 0.0,
        param_mask: Optional[np.ndarray] = None,
        freeze_state_and_post_mlp: bool = False,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"hidden_dim": 256, "cond_dim": 7}

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            d_key,
            c_key,
            share_features_extractor,
            discrete_epsilon,
            param_mask,
            freeze_state_and_post_mlp,
        )

# Alias for compatibility
MlpPolicy = HybridSACPolicy
