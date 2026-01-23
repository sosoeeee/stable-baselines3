# Hybrid SAC (HSAC) Implementation

## Overview

Hybrid SAC is an extension of Soft Actor-Critic (SAC) for environments with **hybrid action spaces** (combination of discrete and continuous actions). This implementation follows the parameterized action MDP framework.

## Algorithm Details

### Action Space

HSAC handles hybrid action spaces defined as:
- **Discrete Action (a)**: Choice of primitive/action type (e.g., "pick", "place", "move")
- **Continuous Parameters (x)**: Parameters for the chosen action (e.g., coordinates, angles)

In Gym, this is represented as a `Dict` space:
```python
action_space = spaces.Dict({
    'id': spaces.Discrete(n),           # Discrete action type
    'params0': spaces.Box(...),          # Parameters for action 0
    'params1': spaces.Box(...),          # Parameters for action 1
    ...
})
```

### Network Architecture

#### Hierarchical Actor
The actor uses a **two-level hierarchy**:

1. **Task Policy (π_task)**: 
   - Input: Observation
   - Output: Categorical distribution over discrete actions
   - Network: MLP → Softmax

2. **Parameter Policy (π_param)**:
   - Input: Observation + One-hot discrete action
   - Output: Gaussian distribution for continuous parameters
   - Network: MLP → Mean + Log_Std (Squashed Gaussian)

#### Hybrid Critic
The critic evaluates state-action pairs:
- Input: Observation + One-hot discrete action + Continuous parameters
- Output: Q-value
- Architecture: Dual Q-networks (like SAC)

### Loss Functions

HSAC uses **dual entropy regularization**:

**Critic Loss:**
```
L_critic = E[(Q(s,a,x) - y)²]
where y = r + γ(1-d)[min_i Q_target(s',a',x') - α_task*log(π_task(a'|s')) - α_param*log(π_param(x'|s',a'))]
```

**Actor Loss:**
```
L_actor = E_s E_a[π_task(a|s) * (α_task*log(π_task(a|s)) + α_param*log(π_param(x|s,a)) - Q(s,a,x))]
```

where:
- `α_task`: Entropy coefficient for discrete actions (auto-tuned)
- `α_param`: Entropy coefficient for continuous parameters (auto-tuned)

## File Structure

```
algs/stable-baselines3/stable_baselines3/
├── hsac/
│   ├── __init__.py           # Module exports
│   ├── hsac.py               # Main HSAC algorithm
│   └── policies.py           # HybridActor, HybridCritic, Policy
├── common/
│   ├── buffers.py            # HybridReplayBuffer added
│   ├── type_aliases.py       # HybridReplayBufferSamples added
│   └── off_policy_algorithm.py  # Modified for Dict actions
└── __init__.py               # Import HSAC

train/
├── hyperparams/
│   └── hsac.yml              # Default hyperparameters
└── train_scripts/
    └── utils.py              # ALGOS dict updated

test/
└── test_hsac.py              # Test suite
```

## Implementation Notes

### Action Space Reorganization
The algorithm internally reorganizes the Gym action space:
- **Input format**: `Dict(id=Discrete(n), params0=Box(...), params1=Box(...), ...)`
- **Internal format**: `Dict(discrete=Discrete(n), continuous=Box(max_dim))`

This allows batch processing with unified parameter dimensions.

### Key Differences from Standard SAC

1. **Dual Entropy Coefficients**: Separate α for discrete and continuous parts
2. **Marginalization in Actor Loss**: Actor loss marginalizes over all discrete actions
3. **Hybrid Replay Buffer**: Stores Dict actions instead of single tensor
4. **Modified Critic**: Takes concatenated (obs, discrete_one_hot, continuous) as input

### Compatible Environments

HSAC works with environments using `spaces.Dict` action spaces:
- gym-hybrid (Moving, Sliding environments)
- Custom environments with parameterized actions
- Any environment following the PAMDP framework

## Hyperparameter Tuning

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ent_coef_task` | `"auto"` | Discrete entropy coefficient (or auto-tune) |
| `ent_coef_param` | `"auto"` | Continuous entropy coefficient (or auto-tune) |
| `target_entropy_task` | `"auto"` | Target entropy for discrete (log(n) by default) |
| `target_entropy_param` | `"auto"` | Target entropy for continuous (-dim by default) |
| `learning_rate` | `3e-4` | Learning rate for all networks |
| `batch_size` | `256` | Minibatch size |
| `tau` | `0.005` | Polyak averaging coefficient |
| `gamma` | `0.99` | Discount factor |

## References

- Original SAC paper: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- Parameterized Action MDPs: [PAMdp Framework](https://arxiv.org/abs/1511.04143)
- Hybrid RL: [P-DQN](https://arxiv.org/abs/1810.06394)

## Citation

If you use this implementation, please cite both Stable-Baselines3 and the original SAC paper.
