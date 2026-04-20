# Atari-MuJoCo-Master

A unified deep reinforcement learning benchmark covering both **discrete control** (Atari ALE) and **continuous control** (MuJoCo), progressing from textbook baselines to a novel hybrid agent that combines distributional RL, latent world-model self-supervision, and adaptive curriculum scheduling.

---

## Project Structure

```
.
├── Policy-Based/          # TD3 for continuous MuJoCo control
│   ├── main.py            # Training entry-point (argparse CLI)
│   ├── td3_agent.py       # TD3 agent (actor + twin critics + target networks)
│   ├── networks.py        # Actor / Critic MLP architectures
│   ├── replay_buffer.py   # Uniform experience replay
│   ├── utils.py           # Evaluation loop
│   ├── plot.py            # Learning-curve visualisation
│   ├── models/            # Saved actor / critic checkpoints
│   └── results/           # Evaluation reward arrays (.npy)
│
└── Value-Based/
    ├── dqn1/              # Dueling Double DQN (strong Atari baseline)
    ├── rainbow1/          # Rainbow-Lite  (Dueling + n-step + PER)
    └── sota/              # Custom hybrid agent (see §Architecture)
        ├── train.py
        ├── core/          # DQN model with distributional head + world model
        ├── networks/      # IMPALA ResNet, MLP, weight initialisers
        ├── utils/         # Prioritised replay, general utilities
        └── results/       # CSV training logs and best-run hyperparameters
```

---

## Algorithms

### Policy-Based — TD3 (MuJoCo)

[Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477) addresses the overestimation bias of DDPG through:

| Trick | Purpose |
|---|---|
| Twin critics | Lower-bound Q target → reduces overestimation |
| Delayed policy update | Every `policy_freq` steps → stabilises training |
| Target policy smoothing | Clipped Gaussian noise on target actions → reduces variance |

**Evaluated environments:** `HalfCheetah-v4` · `Hopper-v4` · `Ant-v4`

---

### Value-Based — Atari ALE

Three implementations show a clean progression of ideas:

| Module | DQN1 | Rainbow-Lite | SOTA |
|---|:---:|:---:|:---:|
| Dueling architecture | ✓ | ✓ | ✓ |
| Double DQN | ✓ | ✓ | ✓ |
| N-step returns | — | ✓ | ✓ (adaptive) |
| Prioritised replay | — | ✓ | ✓ (√-schedule) |
| Distributional RL (C51) | — | — | ✓ |
| IMPALA ResNet encoder | — | — | ✓ |
| Latent world model | — | — | ✓ |
| Random-shift augmentation | — | — | ✓ |
| Dynamic γ scheduling | — | — | ✓ |

---

## Architecture — SOTA Hybrid Agent

```
Observation (96×72 RGB)
        │
        ▼
┌─────────────────────┐
│  IMPALA ResNet      │  ← deep residual CNN (scale_width × 32 channels)
│  + renormalisation  │
└─────────┬───────────┘
          │  latent z  (seq × 32·W × H)
   ┌──────┴──────────────────┐
   │                         │
   ▼                         ▼
Projection MLP          Transition model
(z → 2048-d)            (z, a → ẑ)  ← auxiliary self-supervised loss
   │
   ▼
Dueling C51 head
 ├─ Value  V(s) ─── 51-atom distribution
 └─ Adv A(s,a) ─── per-action 51-atom distribution
          │
          ▼
    Q(s,a) = V + A − mean(A)
```

### Key Design Decisions

1. **IMPALA-style ResNet encoder** — replaces the standard 3-layer CNN; deeper residual blocks give substantially richer visual features with the same frame budget.

2. **Distributional RL (C51, 51 atoms)** — the agent learns the *full return distribution* rather than a scalar Q-value, leading to more stable gradients and better performance on score-heavy games.

3. **MuZero-inspired latent world model** — a convolutional transition model predicts the next latent state given the current state and action.  The cosine-distance reconstruction loss provides a dense, self-supervised training signal at no extra environment cost.

4. **Adaptive n-step returns** — n is annealed from 10 → 3 following a cosine-like curriculum (`n(t) = n₀ · (nf/n₀)^(t/T)`).  High-n at the start encourages long-horizon credit assignment; low-n later reduces variance once the value function is reliable.

5. **Dynamic discount scheduling** — γ warms from 0.97 to 0.997, matching the known benefit of starting with a myopic agent that gradually extends its planning horizon.

6. **Prioritised experience replay with √-scheduling** — priority weights are modulated as `p = TD-error + 0.1 · same_trajectory_mask`, with the maximum priority tracked globally for new transitions.

7. **Random-shift augmentation** — pads each frame by 4 pixels and applies a random integer crop, providing visual invariance at negligible cost.

8. **Decoupled world-model / policy optimisers** — perception + transition parameters form one AdamW group; projection, prediction, and value-head parameters form another.  When the network is periodically hard-reset, optimizer momentum is surgically transplanted to preserve learning-rate warm-up state.

---

## Results

### TD3 — MuJoCo (1 M environment steps)

| Environment | Mean Eval Reward |
|---|---:|
| HalfCheetah-v4 | **~11 000** |
| Hopper-v4 | **~3 500** |
| Ant-v4 | **~5 500** |

### SOTA Agent — Atari ALE

| Game | Score | Human | Random |
|---|---:|---:|---:|
| Pong | **+21** (perfect) | +15 | −18 |
| Boxing | **93.7** | +12 | +0.1 |
| VideoPinball | **482 915** | 17 667 | 0 |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

> MuJoCo 3+ and ALE ROM licences are handled automatically by the `gymnasium` extras.

### Train TD3 (MuJoCo)

```bash
cd Policy-Based
python main.py --env HalfCheetah-v4 --save_model
# Other supported envs: Hopper-v4, Ant-v4
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--env` | `HalfCheetah-v4` | Gymnasium environment ID |
| `--seed` | `0` | Random seed |
| `--max_timesteps` | `1e6` | Total training steps |
| `--start_timesteps` | `25e3` | Pure-exploration warmup |
| `--policy_freq` | `2` | Delayed policy update frequency |
| `--save_model` | off | Persist actor/critic checkpoints |

Plot a saved learning curve:

```bash
python plot.py  # edit ENV_NAME at the top of the file
```

### Train SOTA Agent (Atari)

```bash
cd Value-Based/sota
# Edit env_name at the top of train.py (e.g. 'Pong', 'Boxing', 'VideoPinball')
python train.py
```

Set `QUICK_TEST = True` for a rapid sanity-check run (~3 500 steps).

### Train Baseline DQN / Rainbow-Lite

```bash
cd Value-Based/dqn1   && python DQN1.py      # Dueling Double DQN
cd Value-Based/rainbow1 && python rainbow.py  # Rainbow-Lite
```

---

## Reproducing Saved Results

Pre-trained TD3 checkpoints are stored under `Policy-Based/models/`. To evaluate without retraining:

```bash
cd Policy-Based
python main.py --env HalfCheetah-v4 --load_model default
```

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full pinned list. Core stack:

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended for SOTA agent)
- Gymnasium ≥ 0.29 with `[atari, accept-rom-license]` and `[mujoco]` extras
- einops, tqdm, wandb

---

## References

- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* (TD3), ICML 2018. [[arXiv]](https://arxiv.org/abs/1802.09477)
- Bellemare et al., *A Distributional Perspective on Reinforcement Learning* (C51), ICML 2017. [[arXiv]](https://arxiv.org/abs/1707.06887)
- Hessel et al., *Rainbow: Combining Improvements in Deep RL*, AAAI 2018. [[arXiv]](https://arxiv.org/abs/1710.02298)
- Espeholt et al., *IMPALA: Scalable Distributed Deep-RL*, ICML 2018. [[arXiv]](https://arxiv.org/abs/1802.01561)
- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), Nature 2020. [[arXiv]](https://arxiv.org/abs/1911.08265)
- Wang et al., *Dueling Network Architectures for Deep RL*, ICML 2016. [[arXiv]](https://arxiv.org/abs/1511.06581)
