import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from collections import namedtuple, deque
from itertools import count


import threading
from concurrent.futures import ThreadPoolExecutor

import math


Transition = namedtuple('Transition',
                        ('state', 'reward', 'action', 'c_flag'))

eps=1e-10
_eps = torch.tensor(eps, device='cuda')



class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()

        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (w + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                w + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:w]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)

        eps = 1.0 / (h + 2 * self.pad)
        arange2 = torch.linspace(-1.0 + eps,
                                        1.0 - eps,
                                        h + 2 * self.pad,
                                        device=x.device,
                                        dtype=x.dtype)[:h]
        arange2 = arange2.unsqueeze(1).repeat(1, w).unsqueeze(2)

        base_grid = torch.cat([arange, arange2], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (w + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class PrioritizedReplay_nSteps_Sqrt(object):
    def __init__(self, capacity, total_steps, prefetch_cap=0):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_indices = deque(maxlen=capacity)

        self.priority = torch.zeros(capacity, dtype=torch.float, device='cuda')
        self.max_priority_val = 1.0

        self.total_steps = total_steps

        self.prefetch_cap = prefetch_cap
        self.prefetch_buffer = deque(maxlen=prefetch_cap)
        self.prefetch_buffer_indices = deque(maxlen=prefetch_cap)

        self.aug = RandomShiftsAug(pad=4)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)

    def max_priority(self):
        return self.max_priority_val

    def set_priority(self, indices, priorities, same_traj):
        self.priority[indices] = priorities + 0.1*same_traj
        self.max_priority_val = max(self.max_priority_val, self.priority[indices].max())

    def sample(self, n, batch_size, grad_step):


        alpha = 0.5
        beta = 0.4 + 0.6 * (grad_step / self.total_steps)

        probs = self.priority[:len(self.memory)]**alpha
        probs /= probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=True)


        is_w = (len(self.memory) * probs[indices])**(-1)
        is_w /= is_w.max()



        states_batch, rewards_batch, action_batch, c_flag_batch = [], [], [], []
        next_states_batch = []

        for idx in indices:













            if idx + n >= len(self.memory):
                idx = len(self.memory) - n - 1



            batch = [self.memory[idx+i] for i in range(n)]
            next_state = self.memory[idx+n].state

            states, rewards, actions, c_flags = zip(*batch)

            states_batch.append(torch.stack(states))
            rewards_batch.append(torch.stack(rewards))
            action_batch.append(torch.stack(actions))
            c_flag_batch.append(torch.stack(c_flags))
            next_states_batch.append(next_state)


        states = torch.stack(states_batch).cuda()
        rewards = torch.stack(rewards_batch).cuda()
        action = torch.stack(action_batch).cuda()
        c_flag = torch.stack(c_flag_batch).cuda()
        next_states = torch.stack(next_states_batch).cuda()


        if len(states.shape) == 5:
            b, n, c, h, w = states.shape
            states = self.aug(states.view(-1, c, h, w).float()).view(b, n, c, h, w)
        else:
            states = self.aug(states.float())

        next_states = self.aug(next_states.float())

        return states, next_states, rewards, action, c_flag, indices, is_w

    def free(self):
        self.memory = deque(maxlen=self.capacity)
        self.priority = torch.zeros(self.capacity, dtype=torch.float, device='cuda')
        self.max_priority_val = 1.0
