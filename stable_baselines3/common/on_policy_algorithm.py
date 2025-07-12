import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, HybridRolloutBuffer, HybridDictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, separate_action, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: Union[ActorCriticPolicy]

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: Union[float, Schedule],
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.ent_coef = get_schedule_fn(self.ent_coef)
        self.set_random_seed(self.seed)

        # TODO: when self.action_space is Dict, use DictRolloutBuffer also
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                if isinstance(self.action_space, spaces.Dict):
                    self.rollout_buffer_class = HybridDictRolloutBuffer
                else:
                    self.rollout_buffer_class = DictRolloutBuffer
            elif isinstance(self.action_space, spaces.Dict):
                self.rollout_buffer_class = HybridRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # ======= safety filter to increase the exploration efficiency =======
            # with th.no_grad():
            #     # Convert to pytorch tensor or to TensorDict
            #     obs_tensor = obs_as_tensor(self._last_obs, self.device)
            #     actions, values, log_probs = self.policy(obs_tensor)
              
            # # Convert tensor to numpy, and reshape to the original action shape
            # if isinstance(self.action_space, spaces.Dict):
            #     for key, act in actions.items():
            #         actions[key] = act.cpu().numpy().reshape((-1, *self.action_space.spaces[key].shape))
            # else:
            #     actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

            # # print("actions: ", actions)
            # # print("values: ", values)
            # # print("log_probs: ", log_probs)

            # # Check which actions are valid (implement this function based on your environment)
            # valid_mask, actions_refactor = self._check_action_validity(env, actions)  # Returns boolean array of shape [num_envs]

            # # Resample actions for invalid environments
            # while not np.all(valid_mask):
            #     # Get indices of environments with invalid actions
            #     invalid_indices = np.where(~valid_mask)[0]
                
            #     # Create a sub-observation dictionary for just those environments
            #     invalid_obs = {}
            #     for key, tensor in obs_tensor.items():
            #         invalid_obs[key] = tensor[invalid_indices]
                
            #     # Resample actions just for those environments
            #     with th.no_grad():
            #         new_actions, new_values, new_log_probs = self.policy(invalid_obs)

            #     # Convert to numpy 
            #     if isinstance(self.action_space, spaces.Dict):
            #         for key, act in new_actions.items():
            #             new_actions[key] = act.cpu().numpy().reshape((-1, *self.action_space.spaces[key].shape))
            #     else:
            #         new_actions = new_actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

            #     # print(f"Invalid obs: {invalid_obs}")
            #     # print(f"New actions: {new_actions}")
            #     # print(f"New values: {new_values}")
            #     # print(f"New log_probs: {new_log_probs}")
                
            #     # Replace the invalid actions with the new ones
            #     if isinstance(self.action_space, spaces.Dict):
            #         for i, idx in enumerate(invalid_indices):
            #             values[idx] = new_values[i]
            #             for key in actions.keys():
            #                 actions[key][idx] = new_actions[key][i]
            #                 log_probs[key][idx] = new_log_probs[key][i]
            #     else:
            #         for i, idx in enumerate(invalid_indices):
            #             actions[idx] = new_actions[i]
            #             values[idx] = new_values[i]
            #             log_probs[idx] = new_log_probs[i]
                
            #     # Recheck validity
            #     valid_mask, actions_refactor = self._check_action_validity(env, actions)
            
            # print(f"================= STEP actions =================: {actions_refactor}")
            # print("")

            # new_obs, rewards, dones, infos = env.step(actions_refactor)

            # ====================================================================
            
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            
            # tensor --> numpy
            # if isinstance(self.action_space, spaces.Dict):
            #     for key, act in actions.items():
            #         actions[key] = act.cpu().numpy()
            # else:
            #     actions = actions.cpu().numpy()
            
            # Convert tensor to numpy, and reshape to the original action shape
            if isinstance(self.action_space, spaces.Dict):
                for key, act in actions.items():
                    actions[key] = act.cpu().numpy().reshape((-1, *self.action_space.spaces[key].shape))
            else:
                actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

            # Rescale and perform action
            clipped_actions = actions.copy()

            # TODO: cope with parameterized action space 
            # When coping with parameterized action space, space.Dist has two keys, which are
            # policy.d_key with corresponding space shape of space.Discrete,
            # policy.c_key with corresponding space shape of all discrete actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            elif isinstance(self.action_space, spaces.Dict):
                for key, act in clipped_actions.items():
                    if isinstance(self.action_space.spaces[key], spaces.Box):
                        if self.policy.squash_output:
                            # Rescale to proper domain when using squashing
                            clipped_actions[key] = self.policy.unscale_action(act)  # type: ignore[assignment, arg-type]
                        else:
                            # Actions could be on arbitrary scale, so clip the actions to avoid
                            # out of bound error (e.g. if sampling from a Gaussian distribution)
                            clipped_actions[key] = np.clip(act, self.action_space.spaces[key].low, self.action_space.spaces[key].high)  # type: ignore[assignment, arg-type]

            # TODO: cope with parameterized action space 
            # Convert to array of dictionaries for vectorized environments
            if isinstance(self.action_space, spaces.Dict):
                clipped_actions = [{key: clipped_actions[key][i] for key in clipped_actions} for i in range(len(next(iter(clipped_actions.values()))))]
                # separate the action on parameters dimension
                clipped_actions = separate_action(action=clipped_actions)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # TODO: cope with parameterized action space 
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            elif isinstance(self.action_space, spaces.Dict):
                for key, act in actions.items():
                    if isinstance(self.action_space.spaces[key], spaces.Discrete):
                        actions[key] = act.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self.train() # train before logging

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    # def _check_action_validity(self, env, actions_np):
    #     # print(f"================= _check_action_validity actions_np ===========: {actions_np}")

    #     if isinstance(self.action_space, spaces.Dict):
    #         valid_mask = np.ones(actions_np[next(iter(actions_np))].shape[0], dtype=bool) # get the first key
    #     else:
    #         valid_mask = np.ones(actions_np.shape[0], dtype=bool)

    #     # check if in the valid range
    #     if isinstance(self.action_space, spaces.Box):
    #         if self.policy.squash_output:
    #             # Unscale the actions to match env bounds
    #             # if they were previously squashed (scaled in [-1, 1])
    #             actions_np = self.policy.unscale_action(actions_np)
    #         else:
    #             # Otherwise, clip the actions to avoid out of bound error
    #             # as we are sampling from an unbounded Gaussian distribution
    #             for i in range(actions_np.shape[0]):
    #                 valid_mask[i] = np.all(np.logical_and(actions_np[i] >= self.action_space.low, actions_np[i] <= self.action_space.high))
                    
    #                 # debug print invalid actions
    #                 # if not valid_mask[i]:
    #                 #     print(f"Invalid action at index {i}: {actions_np[i]}")
    #                 #     print(f"Action space low: {self.action_space.low}")
    #                 #     print(f"Action space high: {self.action_space.high}")
                    
    #             # clip for environment check
    #             actions_np = np.clip(actions_np, self.action_space.low, self.action_space.high)
                
    #     elif isinstance(self.action_space, spaces.Dict):
    #         for key, act in actions_np.items():
    #             if isinstance(self.action_space.spaces[key], spaces.Box):
    #                 if self.policy.squash_output:
    #                     # Rescale to proper domain when using squashing
    #                     actions_np[key] = self.policy.unscale_action(act)  # type: ignore[assignment, arg-type]
    #                 else:
    #                     # Actions could be on arbitrary scale, so clip the actions to avoid
    #                     # out of bound error (e.g. if sampling from a Gaussian distribution)
    #                     for i in range(actions_np[key].shape[0]):
    #                         # special case for parameterized action space (skip the action with id=0)
    #                         if actions_np['id'][i] != 0 and valid_mask[i]:
    #                             valid_mask[i] = np.all(np.logical_and(actions_np[key][i] >= self.action_space.spaces[key].low, actions_np[key][i] <= self.action_space.spaces[key].high))
                            
    #                         # debug print invalid actions
    #                         # if not valid_mask[i]:
    #                         #     print(f"Invalid action at index {i}: {actions_np[key][i]}")
    #                         #     print(f"Action space low: {self.action_space.spaces[key].low}")
    #                         #     print(f"Action space high: {self.action_space.spaces[key].high}")

    #                     # clip for environment check
    #                     actions_np[key] = np.clip(act, self.action_space.spaces[key].low, self.action_space.spaces[key].high)


    #     # check if invalid when interacting with the environment
    #     if isinstance(self.action_space, spaces.Dict):
    #         actions_np = [{key: actions_np[key][i] for key in actions_np} for i in range(len(next(iter(actions_np.values()))))]
    #         # separate the action on parameters dimension
    #         actions_refactor = separate_action(action=actions_np)
        
    #     # Use env_method instead of direct attribute call
    #     try:
    #         # For vectorized environments
    #         env_valid_mask = env.env_method_per_env("check_action_validity", actions_refactor)
    #         # Convert from list of results (one per env) to numpy array
    #         env_valid_mask = np.array(env_valid_mask)
    #     except AttributeError:
    #         # Fallback for non-vectorized environments
    #         env_valid_mask = env.env_method("check_action_validity", actions_refactor)

    #     # combine the two masks
    #     for i in range(len(valid_mask)):
    #         valid_mask[i] = valid_mask[i] and env_valid_mask[i]

    #     # DEBUG print invalid actions
    #     if not np.all(valid_mask):
    #         # Convert mask to numpy boolean array if it isn't already
    #         valid_mask_bool = np.array(valid_mask, dtype=bool)
    #         invalid_indices = np.where(~valid_mask_bool)[0]
    #         # print(f"Invalid action indices: {invalid_indices}")
    #         print(f"Invalid actions:")
    #         for idx in invalid_indices:
    #             print(f"  At index {idx}: {actions_refactor[idx]}")
    #         print(f"Valid mask: {valid_mask}")
    #         print("")
        
    #     return valid_mask, actions_refactor