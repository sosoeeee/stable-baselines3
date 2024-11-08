"""Wrapper for rescaling actions to within a max and min action."""
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict
import copy

# Extend the RescaleAction class implemented by the gymnasium library
# to support the dict action space
class RescaleActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleAction
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4")
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1,1,1]))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 0.75])
        >>> wrapped_env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.alltrue(obs == wrapped_env_obs)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray] = -1,
        max_action: Union[float, int, np.ndarray] = 1,
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(env.action_space, Box) or isinstance(env.action_space, Dict), f"expected Box or Dict action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        gym.utils.RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )
        gym.ActionWrapper.__init__(self, env)

        self.action_space = copy.deepcopy(env.action_space)
        if isinstance(self.action_space, Box):
            self.min_action = (
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
            )
            self.max_action = (
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
            )

            self.action_space = Box(
                low=min_action,
                high=max_action,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
            )

        elif isinstance(self.action_space, Dict):
            self.min_action = {}
            self.max_action = {}
            for key in self.action_space.spaces.keys():
                if isinstance(self.action_space.spaces[key], Box):
                    self.min_action[key] = (
                        np.zeros(self.action_space.spaces[key].shape, dtype=self.action_space.spaces[key].dtype) + min_action
                    )
                    self.max_action[key] = (
                        np.zeros(self.action_space.spaces[key].shape, dtype=self.action_space.spaces[key].dtype) + max_action
                    )

                    self.action_space.spaces[key] = Box(
                        low=min_action,
                        high=max_action,
                        shape=env.action_space.spaces[key].shape,
                        dtype=env.action_space.spaces[key].dtype,
                    )

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        if isinstance(self.action_space, Box):
            assert np.all(np.greater_equal(action, self.min_action)), (
                action,
                self.min_action,
            )
            assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
            low = self.env.action_space.low
            high = self.env.action_space.high
            action = low + (high - low) * (
                (action - self.min_action) / (self.max_action - self.min_action)
            )
            action = np.clip(action, low, high)
        elif isinstance(self.action_space, Dict):
            for key in action.keys():
                if isinstance(self.action_space.spaces[key], Box):
                    assert np.all(np.greater_equal(action[key], self.min_action[key])), (
                        action[key],
                        self.min_action[key],
                    )
                    assert np.all(np.less_equal(action[key], self.max_action[key])), (action[key], self.max_action[key])
                    low = self.env.action_space.spaces[key].low
                    high = self.env.action_space.spaces[key].high
                    action[key] = low + (high - low) * (
                        (action[key] - self.min_action[key]) / (self.max_action[key] - self.min_action[key])
                    )
                    action[key] = np.clip(action[key], low, high)

        return action
