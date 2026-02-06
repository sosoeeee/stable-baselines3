from __future__ import annotations

import os
import glob

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import HybridDictReplayBuffer


class DemoBuffer(HybridDictReplayBuffer):
    """
    Demo buffer that extends HybridDictReplayBuffer with the ability to load from .npz files.
    
    Supports both dictionary observation spaces and hybrid action spaces.
    Can load demos from single .npz file or directory of .npz files.
    """

    @classmethod
    def from_npz(
        cls,
        path: str,
        observation_space: spaces.Dict,
        action_space: spaces.Dict,
        device: th.device | None = None,
        n_envs: int = 1,
    ) -> DemoBuffer:
        """
        Load demonstrations from .npz file(s) and create a DemoBuffer.
        
        Args:
            path: Path to a single .npz file or directory containing .npz files
            observation_space: Dictionary observation space
            action_space: Dictionary action space (with 'id' and params keys)
            device: Device to store tensors on (defaults to CPU)
            n_envs: Number of parallel environments
            
        Returns:
            DemoBuffer instance populated with demonstrations
        """
        if device is None:
            device = th.device("cpu")
        
        # Collect all npz files
        if os.path.isfile(path):
            npz_files = [path]
        elif os.path.isdir(path):
            npz_files = sorted(glob.glob(os.path.join(path, "*.npz")))
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory")
        
        if not npz_files:
            raise ValueError(f"No .npz files found in {path}")
        
        # Load all data
        obs_observation_list = []
        obs_achieved_goal_list = []
        obs_desired_goal_list = []
        action_id_list = []
        action_params_list = []
        rewards_list = []
        dones_list = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            
            # Support both new format (dict) and old format (single array)
            if "obs_observation" in data:
                # New format with separate observation components
                obs_observation_list.append(data["obs_observation"])
                obs_achieved_goal_list.append(data["obs_achieved_goal"])
                obs_desired_goal_list.append(data["obs_desired_goal"])
            else:
                raise ValueError(f"File {npz_file} missing observation data")
            
            action_id_list.append(data["action_id"])
            action_params_list.append(data["action_params"])
            
            # Load rewards and dones if available (for compatibility with old files)
            if "rewards" in data:
                rewards_list.append(data["rewards"])
            else:
                raise ValueError(f"File {npz_file} missing rewards data")

            if "dones" in data:
                dones_list.append(data["dones"])
            else:
                raise ValueError(f"File {npz_file} missing dones data")

        # Concatenate all trajectories
        obs_observation = np.concatenate(obs_observation_list, axis=0)
        obs_achieved_goal = np.concatenate(obs_achieved_goal_list, axis=0)
        obs_desired_goal = np.concatenate(obs_desired_goal_list, axis=0)
        action_ids = np.concatenate(action_id_list, axis=0)
        action_params = np.concatenate(action_params_list, axis=0)
        rewards = np.concatenate(rewards_list, axis=0)
        dones = np.concatenate(dones_list, axis=0)
        
        # Create buffer instance
        buffer_size = len(obs_observation)
        demo_buffer = cls(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        
        # Populate buffer with demonstrations
        for i in range(buffer_size):
            # Create observation dict
            obs = {
                "observation": obs_observation[i],
                "achieved_goal": obs_achieved_goal[i],
                "desired_goal": obs_desired_goal[i],
            }
            
            # Create action dict
            action = {"id": action_ids[i]}
            # Add all param components to action dict
            param_start = 0
            for key in sorted(action_space.spaces.keys()):
                if key == "id":
                    continue
                param_size = action_space[key].shape[0]
                action[key] = action_params[i, param_start:param_start + param_size]
                param_start += param_size
            
            # Dummy next_obs, reward, done (not used for demo buffer)
            next_obs = obs.copy()
            reward = np.array([rewards[i]])
            done = np.array([dones[i]])
            infos = [{}]
            
            demo_buffer.add(obs, next_obs, action, reward, done, infos)
        
        return demo_buffer
