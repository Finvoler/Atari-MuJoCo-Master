import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import ale_py
import os
import pandas as pd
from gymnasium.wrappers import AtariPreprocessing, FrameStack




ENV_ID = "ALE/Breakout-v5"
SEED = 42

TOTAL_TIMESTEPS = 10_000_000

LEARNING_RATE = 1e-4


BUFFER_SIZE = 50_000
LEARNING_STARTS = 25_000
BATCH_SIZE = 32
GAMMA = 0.99


EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 1_000_000

TRAIN_FREQ = 4
TARGET_UPDATE_FREQ = 10_000

LOG_FILENAME = "training_log_rainbow_adam.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")




class FireResetEnv(gym.Wrapper):
    """
    Automatically presses the FIRE action on each environment reset.
    Required for games like Breakout where the ball won't launch otherwise.
    """
    def __init__(self, env):
        super().__init__(env)

        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)


        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
            return obs, {}


        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
            return obs, {}


        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)

        return obs, {}

    def step(self, ac):
        return self.env.step(ac)




def make_env(env_id, seed):
    env = gym.make(env_id, render_mode="rgb_array", frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,

        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False
    )


    env = FireResetEnv(env)

    env = FrameStack(env, num_stack=4)
    env.action_space.seed(seed)
    return env




class DuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        input_dim = 64 * 7 * 7

        self.advantage = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.float() / 255.0
        feat = self.features(x)
        val = self.value(feat)
        adv = self.advantage(feat)
        return val + (adv - adv.mean(dim=1, keepdim=True))




class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state)).to(device),
            torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_state)).to(device),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)




if __name__ == "__main__":
    env = make_env(ENV_ID, SEED)

    q_network = DuelingDQN(env.action_space.n).to(device)
    target_network = DuelingDQN(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())


    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE, eps=1e-8)

    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(BUFFER_SIZE)

    log_data = []

    obs, _ = env.reset(seed=SEED)
    obs = np.array(obs)

    episode_reward = 0
    print("Starting Rainbow Lite (Adam + FireReset)...")

    for global_step in range(TOTAL_TIMESTEPS):
        epsilon = np.interp(global_step, [0, EPS_DECAY_STEPS], [EPS_START, EPS_END])

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
                q_values = q_network(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        real_reward = np.clip(reward, -1, 1)
        real_next_obs = np.array(next_obs)
        buffer.push(obs, action, real_reward, real_next_obs, done)
        obs = real_next_obs

        if done:
            if len(log_data) % 10 == 0:
                print(f"Step {global_step}: Episode Finished. Reward: {episode_reward}, Eps: {epsilon:.3f}")

            log_data.append({"step": global_step, "reward": episode_reward})

            if len(log_data) % 100 == 0:
                pd.DataFrame(log_data).to_csv(LOG_FILENAME, index=False)

            episode_reward = 0
            obs, _ = env.reset()
            obs = np.array(obs)

        if global_step > LEARNING_STARTS and global_step % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)


            with torch.no_grad():
                best_actions = q_network(next_states).argmax(dim=1).unsqueeze(1)
                target_values = target_network(next_states).gather(1, best_actions)
                td_target = rewards + (GAMMA * target_values * (1 - dones))

            current_q = q_network(states).gather(1, actions)
            loss = loss_fn(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(q_network.state_dict())
            print(f"Step {global_step}: Target Network Updated.")

    pd.DataFrame(log_data).to_csv(LOG_FILENAME, index=False)
    torch.save(q_network.state_dict(), "rainbow_adam_breakout.pth")
    env.close()
    print("Training Finished!")