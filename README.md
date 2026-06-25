# ReLAB

<img src="./media/snake.gif" width="400"/>

Gymnasium-based environment for RL research/experiments 

---

### Observation space

The state returned after each `.reset()` or `.step()` is an 11-dimensional binary vector:

```
[danger_straight, danger_right, danger_left,
 dir_up, dir_down, dir_left, dir_right,
 food_up, food_down, food_left, food_right]
```

All values are `0` or `1`. The agent never sees the raw grid — only relative danger, heading, and food direction. This keeps the state space small and interpretable, at the cost of long-range planning: an agent trained on this vector plays well early but struggles when the body grows long enough to trap itself.

---

### Action space

Discrete(4): `UP=0  DOWN=1  LEFT=2  RIGHT=3`

Attempting to reverse direction (e.g. LEFT while heading RIGHT) is silently ignored — the snake continues in its current direction. The environment handles this internally so the agent can submit any of the four actions without crashing.

---

### Horizon

The `max_steps` parameter (default `1000`) controls the number of steps after which `truncated` is set to `True` and the episode ends. Without it, an untrained agent can loop indefinitely without dying, stalling training. Reduce it for faster early learning, increase it once the agent can reliably find food.

```python
env = SnakeEnv(grid_size=10, max_steps=500)
```

---

### Reward

**Default reward structure:**

| Event | Reward |
|-------|--------|
| Eat food | `+1.0` |
| Die (wall or body) | `-1.0` |
| Each step | `-0.01` (configurable) |

The step penalty matters. Without it, a policy that survives without eating is locally optimal — it never dies, so it never receives `-1`, and it never needs to seek food. The small negative per step forces the agent to prefer eating over stalling.

Reward shaping is exposed through `EnvConfig`:

```python
from configs.config import EnvConfig
env = SnakeEnv(**EnvConfig(step_penalty=0.0).__dict__)  # disable step penalty
```

---

### Agents

**Q-Learning (tabular)**

Stores a dictionary `{state_tuple → [Q(s,a0)...Q(s,a3)]}`. Updates with the standard TD(0) rule:

```
Q(s,a) ← Q(s,a) + α [r + γ · max Q(s',a') - Q(s,a)]
```

Works well on this environment because the observation is already binary — the state space is at most `2^11 = 2048` entries.

```bash
python run_train.py --agent qlearning --episodes 2000
```

**DQN**

Replaces the Q-table with a two-layer MLP `(11 → 128 → 128 → 4)`. Three stabilizing mechanisms:

- **Replay buffer** — stores past transitions, breaks temporal correlations between updates
- **Target network** — a frozen copy of the policy net, synced every 500 gradient steps, provides stable Q-targets
- **Huber loss** — less sensitive to large TD errors than MSE during early chaotic training

```bash
python run_train.py --agent dqn --episodes 2000
```

---

### Training

```bash
# DQN with defaults
python run_train.py

# Q-Learning, larger grid, fixed seed
python run_train.py --agent qlearning --grid 15 --seed 0

# Custom output directory
python run_train.py --exp experiments/my_run
```

Each run saves to its experiment directory:

```
experiments/my_run/
├── train_log.csv          # episode, reward, score, epsilon, loss per step
├── checkpoint_best.pt     # best checkpoint by eval reward
├── checkpoint_ep500.pt    # periodic checkpoint
└── checkpoint_final.pt    # last episode
```
