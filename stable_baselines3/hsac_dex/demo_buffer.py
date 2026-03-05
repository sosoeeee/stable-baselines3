from __future__ import annotations

import os
import glob

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.her.her_replay_buffer import HybridHerReplayBuffer


class DemoBuffer(HybridHerReplayBuffer):
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
        **her_kwargs,
    ) -> DemoBuffer:
        """
        Load demonstrations from .npz file(s) and create a DemoBuffer.
        
        Args:
            path: Path to a single .npz file or directory containing .npz files
            observation_space: Dictionary observation space
            action_space: Dictionary action space (with 'id' and params keys)
            device: Device to store tensors on (defaults to CPU)
            **her_kwargs: Additional arguments for HybridHerReplayBuffer (env, n_sampled_goal, etc.)
            
        Returns:
            DemoBuffer instance populated with demonstrations
        """
        n_envs = 1  # Demo buffer always uses single environment
        if device is None:
            device = th.device("cpu")
        if "env" not in her_kwargs:
            raise ValueError(
                "DemoBuffer.from_npz() requires `env` in her_kwargs because "
                "HybridHerReplayBuffer needs it for HER reward computation."
            )
        
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
            
            if "obs_observation" in data:
                obs_observation_list.append(data["obs_observation"])
                obs_achieved_goal_list.append(data["obs_achieved_goal"])
                obs_desired_goal_list.append(data["obs_desired_goal"])
            else:
                raise ValueError(f"File {npz_file} missing observation data")
            
            action_id_list.append(data["action_id"])
            action_params_list.append(data["action_params"])
            
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
        
        total_steps = len(dones)
        buffer_size = total_steps
        
        print("\n=== Demo Buffer Loading Info ===")
        print(f"Total steps: {total_steps}")
        print(f"Buffer size: {buffer_size}")
        print("================================\n")
        
        # Create buffer instance with n_envs=1
        demo_buffer = cls(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            **her_kwargs,
        )
        
        # Load all steps sequentially using add() to maintain HER metadata
        for step_idx in range(total_steps):
            # Prepare next_obs dict
            if step_idx < total_steps - 1 and not dones[step_idx]:
                next_step_idx = step_idx + 1
            else:
                next_step_idx = step_idx
            
            # Prepare data in (n_envs, ...) format where n_envs=1
            obs = {
                "observation": obs_observation[step_idx:step_idx+1],
                "achieved_goal": obs_achieved_goal[step_idx:step_idx+1],
                "desired_goal": obs_desired_goal[step_idx:step_idx+1],
            }
            
            next_obs = {
                "observation": obs_observation[next_step_idx:next_step_idx+1],
                "achieved_goal": obs_achieved_goal[next_step_idx:next_step_idx+1],
                "desired_goal": obs_desired_goal[next_step_idx:next_step_idx+1],
            }
            
            action = {
                "discrete": action_ids[step_idx:step_idx+1],
                "continuous": action_params[step_idx:step_idx+1],
            }
            
            reward = rewards[step_idx:step_idx+1]
            done = dones[step_idx:step_idx+1]
            infos = [{}]  # Single environment, single info dict
            
            # Call parent's add method
            demo_buffer.add(obs, next_obs, action, reward, done, infos)
        
        print(f"✓ Loaded {total_steps} steps")
        print(f"  Buffer pos: {demo_buffer.pos}, full: {demo_buffer.full}\n")
        
        return demo_buffer
