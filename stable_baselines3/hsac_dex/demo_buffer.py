from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch as th


@dataclass
class DemoBuffer:
    observations: np.ndarray
    action_ids: np.ndarray
    action_params: np.ndarray
    device: th.device

    @classmethod
    def from_npz(cls, path: str, device: th.device) -> "DemoBuffer":
        data = np.load(path)
        observations = data["obs"]
        action_ids = data["action_id"]
        action_params = data["action_params"]
        return cls(observations, action_ids, action_params, device)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        max_idx = self.observations.shape[0]
        batch_size = min(batch_size, max_idx)
        indices = np.random.randint(0, max_idx, size=batch_size)
        obs = self.observations[indices]
        action_ids = self.action_ids[indices]
        action_params = self.action_params[indices]
        return obs, {"id": action_ids, "params": action_params}