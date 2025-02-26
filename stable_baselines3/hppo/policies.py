import collections
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial
import warnings

import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    HybridActionMlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule, TensorDict

class HybridActorCriticPolicy(BasePolicy):
    """
    Policy class for hybrid actor-critic algorithms (has both policy and value prediction).
    Used by H-PPO.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            d_key: str = 'id',
            c_key: str = 'params',
            mask: Optional[np.ndarray] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
                squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        # TODO: Two distributions for discrete and continuous actions
        assert isinstance(action_space, spaces.Dict)
        self.d_key = d_key
        self.c_key = c_key
        self.mask = mask
        self.action_dist = {}
        self.action_dist[self.d_key] = make_proba_distribution(self.action_space[self.d_key], use_sde=use_sde,
                                                               dist_kwargs=dist_kwargs)
        self.action_dist[self.c_key] = make_proba_distribution(self.action_space[self.c_key], use_sde=use_sde,
                                                               dist_kwargs=dist_kwargs)
        if self.d_key is None or self.c_key is None:
            raise NotImplementedError(
                "Parameterized action space should at least have one discrete action space for action types and one continuous action space for action parameters.")

        self._build(lr_schedule)

    def get_d_key(self) -> str:
        return self.d_key

    def get_c_key(self) -> str:
        return self.c_key

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(
            lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    # def reset_noise(self, n_envs: int = 1) -> None:
    #     """
    #     Sample new weights for the exploration matrix.

    #     :param n_envs:
    #     """
    #     assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
    #     self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks and the features extractor for discrete and continuous actions.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        self.action_features_extractor = HybridActionMlpExtractor(
            self.mlp_extractor.latent_dim_pi,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        last_dim_di = self.action_features_extractor.latent_dim_di
        last_dim_co = self.action_features_extractor.latent_dim_co

        # TODO: create two types of action net
        self.action_net = {}
        self.action_net[self.d_key] = self.action_dist[self.d_key].proba_distribution_net(latent_dim=last_dim_di)
        self.action_net[self.c_key], self.log_std = self.action_dist[self.c_key].proba_distribution_net(latent_dim=last_dim_co, log_std_init=self.log_std_init)
        self.action_net = nn.ModuleDict(self.action_net)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_features_extractor: np.sqrt(2),     # TODO: check for action_features_extractor
                self.action_net[self.d_key]: 0.01,
                self.action_net[self.c_key]: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[TensorDict, th.Tensor, TensorDict]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        # TODO: Two types of distribution, and joint them
        actions = {}
        log_prob = {}

        dis_distribution = self._get_dis_action_dist_from_latent(latent_pi)
        actions[self.d_key] = dis_distribution.get_actions(deterministic=deterministic)
        log_prob[self.d_key] = dis_distribution.log_prob(actions[self.d_key])

        con_distribution = self._get_con_action_dist_from_latent(latent_pi, actions[self.d_key])
        actions[self.c_key] = con_distribution.get_actions(deterministic=deterministic)
        log_prob[self.c_key] = con_distribution.log_prob(actions[self.c_key])

        actions[self.d_key] = actions[self.d_key].reshape((-1, *self.action_space.spaces[self.d_key].shape))
        actions[self.c_key] = actions[self.c_key].reshape((-1, *self.action_space.spaces[self.c_key].shape))

        return actions, values, log_prob

    def extract_features(  # type: ignore[override]
            self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs,
                                            self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_dis_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve discrete action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Discrete action distribution
        """
        latent_di = self.action_features_extractor.forward_discrete(latent_pi) # add a independent feature extractor for discrete action
        mean_actions = self.action_net[self.d_key](latent_di)
        distribution = self.action_dist[self.d_key].proba_distribution(action_logits=mean_actions)

        return distribution

    def _get_con_action_dist_from_latent(self, latent_pi: th.Tensor, action_type: th.Tensor) -> Distribution:
        """
        Retrieve continuous action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param action_type: Discrete action
        :return: Continuous action distribution
        """
        latent_co = self.action_features_extractor.forward_continuous(latent_pi) # add a independent feature extractor for continuous action
        mean_actions = self.action_net[self.c_key](latent_co)

        with th.no_grad():
            action_type_np = action_type.cpu().numpy()
            mask_tensor = self.mask[action_type_np]

        # covert to tensor
        mask_tensor = th.tensor(mask_tensor, device=mean_actions.device, dtype=mean_actions.dtype)

        masked_mean_actions = mean_actions * mask_tensor
        masked_log_std = self.log_std * mask_tensor

        distribution = self.action_dist[self.c_key].proba_distribution(masked_mean_actions, masked_log_std)

        return distribution

    # TODO: predict discrete action and its' parameters respectively, and joint them
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> TensorDict:
        """
        Get the action according to the policy for a given observation.
        This is used for evaluation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_action(observation, deterministic)

    # TODO: evaluate discrete action and its' parameters respectively
    def evaluate_actions(self, obs: PyTorchObs, actions: TensorDict) -> Tuple[
        th.Tensor, TensorDict, Optional[TensorDict]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)

        log_prob = {}
        entropy = {}

        dis_distribution = self._get_dis_action_dist_from_latent(latent_pi)
        log_prob[self.d_key] = dis_distribution.log_prob(actions[self.d_key])
        entropy[self.d_key] = dis_distribution.entropy()

        con_distribution = self._get_con_action_dist_from_latent(latent_pi, actions[self.d_key])
        log_prob[self.c_key] = con_distribution.log_prob(actions[self.c_key])
        entropy[self.c_key] = con_distribution.entropy()

        return values, log_prob, entropy

    # TODO: return two types of distribution
    def get_action(self, obs: PyTorchObs, deterministic) -> TensorDict:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)

        actions = {}
        dis_distribution = self._get_dis_action_dist_from_latent(latent_pi)
        actions[self.d_key] = dis_distribution.get_actions(deterministic=deterministic)
        con_distribution = self._get_con_action_dist_from_latent(latent_pi, actions[self.d_key])
        actions[self.c_key] = con_distribution.get_actions(deterministic=deterministic)

        return actions

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


class HybridActorCriticCnnPolicy(HybridActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            d_key: str = 'id',
            c_key: str = 'params',
            mask: Optional[np.ndarray] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            d_key,
            c_key,
            mask,
        )


class MultiInputHybridActorCriticPolicy(HybridActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            d_key: str = 'id',
            c_key: str = 'params',
            mask: Optional[np.ndarray] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            d_key,
            c_key,
            mask,
        )


MlpPolicy = HybridActorCriticPolicy
CnnPolicy = HybridActorCriticCnnPolicy
MultiInputPolicy = MultiInputHybridActorCriticPolicy