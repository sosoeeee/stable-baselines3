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
        
        新策略：将完整的 episodes 轮流分配到不同的环境中
        - 例如：n_envs=2, 3个episodes → env0放ep 1,3; env1放ep 2
        - 保持每个 episode 的完整性
        - 更好地利用并行环境结构
        
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
            
            # Load rewards and dones if available
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
        
        # 根据 dones 分割 episodes
        episodes = []
        current_episode_indices = []
        for i in range(len(dones)):
            current_episode_indices.append(i)
            if dones[i]:
                episodes.append(current_episode_indices)
                current_episode_indices = []
        # 如果最后有未完成的 episode（不应该发生，但以防万一）
        if current_episode_indices:
            print(f"Warning: Found incomplete episode at the end with {len(current_episode_indices)} steps")
            episodes.append(current_episode_indices)
        
        num_episodes = len(episodes)
        max_episode_length = max(len(ep) for ep in episodes)
        
        print(f"\n=== Demo Buffer Loading Info ===")
        print(f"Total steps: {len(dones)}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Episode lengths: min={min(len(ep) for ep in episodes)}, "
              f"max={max_episode_length}, avg={np.mean([len(ep) for ep in episodes]):.1f}")
        print(f"Target n_envs: {n_envs}")
        
        # 计算 buffer_size：每个环境需要容纳的最大步数
        # episodes_per_env = ceil(num_episodes / n_envs)
        episodes_per_env = (num_episodes + n_envs - 1) // n_envs
        buffer_size = episodes_per_env * max_episode_length
        
        print(f"Episodes per env: {episodes_per_env}")
        print(f"Buffer size per env: {buffer_size}")
        print(f"Total buffer capacity: {buffer_size * n_envs}")
        print(f"================================\n")
        
        # Create buffer instance
        demo_buffer = cls(
            buffer_size=buffer_size * n_envs,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        
        # 将 episodes 轮流分配到不同环境中
        # episode 0 -> env 0, episode 1 -> env 1, ..., episode n_envs -> env 0, ...
        env_positions = [0] * n_envs  # 记录每个环境当前写入位置
        
        for ep_idx, episode_indices in enumerate(episodes):
            env_idx = ep_idx % n_envs  # 轮流分配
            pos = env_positions[env_idx]
            
            # 将这个 episode 的所有步骤写入对应环境
            for step_idx in episode_indices:
                # 直接操作内部数组
                for key in demo_buffer.observations.keys():
                    demo_buffer.observations[key][pos, env_idx] = obs_observation[step_idx] if key == "observation" \
                        else obs_achieved_goal[step_idx] if key == "achieved_goal" \
                        else obs_desired_goal[step_idx]
                    # next_obs 使用下一步的 obs（如果是最后一步则使用当前 obs）
                    next_step_idx = step_idx + 1 if step_idx + 1 < len(dones) and not dones[step_idx] else step_idx
                    demo_buffer.next_observations[key][pos, env_idx] = \
                        obs_observation[next_step_idx] if key == "observation" \
                        else obs_achieved_goal[next_step_idx] if key == "achieved_goal" \
                        else obs_desired_goal[next_step_idx]
                
                # 为所有 action keys 写入数据
                demo_buffer.actions["discrete"][pos, env_idx] = int(action_ids[step_idx])
                demo_buffer.actions["continuous"][pos, env_idx] = action_params[step_idx]
                
                demo_buffer.rewards[pos, env_idx] = rewards[step_idx]
                demo_buffer.dones[pos, env_idx] = dones[step_idx]
                
                pos += 1
            
            # 更新该环境的写入位置
            env_positions[env_idx] = pos
        
        # 设置 buffer 的 pos 和 full 标志
        demo_buffer.pos = min(env_positions)
        demo_buffer.full = (demo_buffer.pos >= buffer_size)
        
        print(f"✓ Loaded {num_episodes} episodes into {n_envs} environment(s)")
        print(f"  Env positions: {env_positions}")
        print(f"  Buffer pos: {demo_buffer.pos}, full: {demo_buffer.full}\n")
        
        return demo_buffer
