import gymnasium as gym
import ale_py
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, chain
import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import torchvision.transforms as transforms
import locale
import os

from utils.general import Hypers, add_to_csv, params_count, params_and_grad_norm, seed_np_torch
from utils.experience_replay import PrioritizedReplay_nSteps_Sqrt, Transition
from core.dqn_model import DQN

locale.getpreferredencoding = lambda: "UTF-8"


#  Configure game environment below
env_name = 'Boxing'
SEED = 8231

# Set to True for a quick end-to-end sanity check
QUICK_TEST = False

batch_size = 8 if QUICK_TEST else 32
lr=1e-4
eps=1e-8

critic_ema_decay=0.995

initial_gamma=torch.tensor(1-0.97).log()
final_gamma=torch.tensor(1-0.997).log()

initial_n = 10
final_n = 3

num_buckets=51

reset_every = 800 if QUICK_TEST else 40000 
schedule_max_step = 100 if QUICK_TEST else reset_every//4 
total_steps = 3500 if QUICK_TEST else 102000 

prefetch_cap=1

# Enable checkpoint resumption from the last saved state
RESUME = False
RESUME_PATH = 'checkpoints/atari_last.pth'

memory = PrioritizedReplay_nSteps_Sqrt(total_steps+5, total_steps=schedule_max_step, prefetch_cap=prefetch_cap)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
def save_checkpoint(net, model_target, optimizer, step, grad_step, path):
    torch.save({
            'model_state_dict': net.state_dict(),
            'model_target_state_dict': model_target.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'grad_step': grad_step,
            }, path)

class MaxLast2FrameSkipWrapper(Hypers, gym.Wrapper):
    def __init__(self, env, skip=4, noops=30, seed=0):
        super().__init__(env=env)
        self.env.action_space.seed(seed)
        
    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)

        return obs, _
        
    def noop_steps(self, states):
        noops = random.randint(0,self.noops)
        
        for i in range(noops):
            state = self.step(np.array([0]))[0]
            state = preprocess(state)
            states.append(state)
        return states

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward

            terminated = np.logical_or(done, truncated)
            if done or truncated:
                break
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info

def make_env():
    env = gym.make(f"ALE/{env_name}-v5", frameskip=1, repeat_action_probability=0.0)
    env = MaxLast2FrameSkipWrapper(env, seed=SEED)
    return env

env = gym.vector.SyncVectorEnv([make_env])

n_actions = env.action_space[0].n

state, info = env.reset()
n_observations = len(state)

seed_np_torch(SEED)

def copy_states(source, target):
    for key, _ in zip(source.state_dict()['state'].keys(), target.state_dict()['state'].keys()):

        target.state_dict()['state'][key]['exp_avg_sq'] = copy.deepcopy(source.state_dict()['state'][key]['exp_avg_sq'])
        target.state_dict()['state'][key]['exp_avg'] = copy.deepcopy(source.state_dict()['state'][key]['exp_avg'])
        target.state_dict()['state'][key]['step'] = copy.deepcopy(source.state_dict()['state'][key]['step'])
        
def target_model_ema(model, model_target):
    with torch.no_grad():
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data = critic_ema_decay * param_target.data + (1.0 - critic_ema_decay) * param.data.clone()

model=DQN(n_actions, num_buckets=num_buckets).cuda()
model_target=DQN(n_actions, num_buckets=num_buckets).cuda()

model_target.load_state_dict(model.state_dict())

perception_modules=[model.encoder_cnn, model.transition]
actor_modules=[model.prediction, model.projection, model.a, model.v]

params_wm=[]
for module in perception_modules:
    for param in module.parameters():
        if param.requires_grad==True: 
            params_wm.append(param)

params_ac=[]
for module in actor_modules:
    for param in module.parameters():
        if param.requires_grad==True:
            params_ac.append(param)


optimizer = torch.optim.AdamW(chain(params_wm, params_ac),
                                lr=lr, weight_decay=0.1, eps=1.5e-4)

train_tfms = transforms.Compose([
                         transforms.Resize((96,72)),
                        ])


def preprocess(state):
    state=torch.tensor(state, dtype=torch.float, device='cuda') / 255
    state=train_tfms(state.permute(0,3,1,2))
    return state.squeeze(0)

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


def epsilon_greedy(Q_action, step, final_eps=0, num_envs=1):
    epsilon = linearly_decaying_epsilon(2001, step, 2000, final_eps)
    
    if random.random() < epsilon:
        action = torch.randint(0, n_actions, (num_envs,), dtype=torch.int64, device='cuda').squeeze(0)
    else:
        action = Q_action.view(num_envs).squeeze(0).to(torch.int64)
    return action

def project_distribution(supports, weights, target_support):
    with torch.no_grad():
        v_min, v_max = target_support[0], target_support[-1]
        num_dims = target_support.shape[-1]
        delta_z = (v_max - v_min) / (num_buckets-1)
        clipped_support = supports.clip(v_min, v_max)
        numerator = (clipped_support[:,None] - target_support[None,:,None].repeat_interleave(clipped_support.shape[0],0)).abs()
        quotient = 1 - (numerator / delta_z)
        clipped_quotient = quotient.clip(0, 1)
        inner_prod = (clipped_quotient * weights[:,None]).sum(-1)
        return inner_prod.squeeze()


mse = torch.nn.MSELoss(reduction='none')


try:
    scaler = torch.amp.GradScaler('cuda') 
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler() 
def optimize(step, grad_step, n):
        
    model.train()
    model_target.train()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
        with torch.no_grad():
            states, next_states, rewards, action, c_flag, idxs, is_w = memory.sample(n, batch_size, grad_step)
        terminal=1-c_flag
        
        available_frames = min(n - 1, states.shape[1] - 1) 
        transition_len = min(5, available_frames) if available_frames > 0 else 0
        
        if transition_len > 0:
            with torch.no_grad():
                z = model_target.encode(states[:, 1:1+transition_len])[0]
            q, max_action, _, z_pred = model(states[:,0][:,None], action[:, :transition_len].long())
        else:
            z = None
            z_pred = None
            encoded, _ = model.encode(states[:,0][:,None])
            q, max_action = model.q_head(encoded)
        

        
        max_action  = model.get_max_action(next_states[:,None])
        next_values = model_target.evaluate(next_states[:,None].contiguous(), max_action)
        

        action = action[:,0,None].expand(batch_size,num_buckets)
        action=action[:,None]
        with torch.no_grad():
            gammas_one=torch.ones(batch_size,n,1,dtype=torch.float,device='cuda')
            gamma_val = ( (schedule_max_step - min(grad_step, schedule_max_step)) / schedule_max_step) * (initial_gamma-final_gamma) + final_gamma
            gamma_step = 1-gamma_val.exp()
            gammas=gammas_one*gamma_step

            
            returns = []
            for t in range(n):
                ret = 0
                for u in reversed(range(t, n)):
                    ret += torch.prod(c_flag[:,t+1:u+1],-2)*torch.prod(gammas[:,t:u],-2)*rewards[:,u]
                returns.append(ret)
            returns = torch.stack(returns,1)
        
        plot_vs = returns.clone().sum(-1)
        
        same_traj = (torch.prod(c_flag[:,:n],-2)).squeeze()
        
        returns = returns[:,0]
        returns = returns + torch.prod(gammas[0,:10],-2).squeeze()*same_traj[:,None]*model.support[None,:]
        returns = returns.squeeze()
        
        next_values = next_values[:,0]

        log_probs = torch.log(q[:,0].gather(-2, action)[:,None] + eps).contiguous()
        
        
        dist = project_distribution(returns, next_values.squeeze(), model.support)
        
        loss = -(dist*(log_probs.squeeze())).sum(-1).view(batch_size,-1).sum(-1)
        dqn_loss = loss.clone().mean()
        td_error = (loss + torch.nan_to_num((dist*torch.log(dist))).sum(-1)).mean()

        
        batched_loss = loss.clone()
        
        
        if z is not None and z_pred is not None:
            z = F.normalize(z, 2, dim=-1, eps=1e-5)
            z_pred = F.normalize(z_pred, 2, dim=-1, eps=1e-5)
            recon_loss = (mse(z_pred.contiguous().view(-1,2048), z.contiguous().view(-1,2048))).sum(-1)
            recon_loss = 5*(recon_loss.view(batch_size, -1).mean(-1))*same_traj
        else:
            recon_loss = torch.zeros(batch_size, device='cuda')
        
        
        loss += recon_loss
        
        loss = (loss*is_w).mean() 

    loss.backward()

    param_norm, grad_norm = params_and_grad_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
    
    optimizer.step()
    optimizer.zero_grad()
    
    memory.set_priority(idxs, batched_loss, same_traj)
    
    
    lr = optimizer.param_groups[0]['lr']

start_step = 0
start_grad_step = 0
if RESUME and os.path.exists(RESUME_PATH):
    try:
        print(f"Loading checkpoint from {RESUME_PATH}...")
        checkpoint = torch.load(RESUME_PATH, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_target.load_state_dict(checkpoint['model_target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        start_grad_step = checkpoint.get('grad_step', 0)
        print(f"Resumed from step {start_step}")
    except Exception as e:
        print(f"Checkpoint load failed: {e}, starting from scratch.")

scores=[]
memory.free()
step=start_step
grad_step=start_grad_step

progress_bar = tqdm.tqdm(total=total_steps, initial=start_step)

while step<(total_steps):
    state, info = env.reset()
    state = preprocess(state)

    states = deque(maxlen=4)
    for i in range(4):
        states.append(state)
    
    
    eps_reward=torch.tensor([0], dtype=torch.float)
    
    reward=np.array([0])
    done_flag=np.array([False])
    terminated=np.array([False])

    last_lives=np.array([0])
    life_loss=np.array([0])
    resetted=np.array([0])
    
    last_grad_update=0
    while step<(total_steps):
        progress_bar.update(1)
        model_target.train()
        
        len_memory = len(memory)
        
        Q_action = model_target.env_step(torch.cat(list(states),-3).unsqueeze(0).unsqueeze(0))
        
        action = epsilon_greedy(Q_action, len_memory).cpu()
        
        memory.push(torch.cat(list(states),-3).detach().cpu(), torch.tensor(reward,dtype=torch.float), action,
                    torch.tensor(1 - np.logical_or(done_flag, life_loss),dtype=torch.float))
        
        state, reward, terminated, truncated, info = env.step([action.numpy()])
        state = preprocess(state)
        states.append(state)
        
        eps_reward+=reward
        reward = reward.clip(-1, 1)


        
        done_flag = np.logical_or(terminated, truncated)
        lives = info['lives']
        life_loss = (last_lives-lives).clip(min=0)
        resetted = (lives-last_lives).clip(min=0)
        last_lives = lives

        
        n = int(initial_n * (final_n/initial_n)**(min(grad_step,schedule_max_step) / schedule_max_step))
        n = np.array(n).item()
        
        memory.priority[len_memory] = memory.max_priority()
        

        if len_memory>2000:
            for i in range(2):
                optimize(step, grad_step, n)
                target_model_ema(model, model_target)
                grad_step+=1

        
        if ((step+1)%10000)==0:
            save_checkpoint(model, model_target, optimizer, step, grad_step, 'checkpoints/atari_last.pth')
        
            
        
        if grad_step>reset_every:
            print('Reseting on step', step, grad_step)
            
            random_model = DQN(n_actions, num_buckets=num_buckets).cuda()
            model.hard_reset(random_model)
            
            random_model = DQN(n_actions, num_buckets=num_buckets).cuda()
            model_target.hard_reset(random_model)
            seed_np_torch(SEED)
            
            random_model=None
            grad_step=0

            actor_modules=[model.prediction, model.projection, model.a, model.v]
            params_ac=[]
            for module in actor_modules:
                for param in module.parameters():
                    params_ac.append(param)
                    

            perception_modules=[model.encoder_cnn, model.transition]
            params_wm=[]
            for module in perception_modules:
                for param in module.parameters():
                    params_wm.append(param)
            
            optimizer_aux = torch.optim.AdamW(params_wm, lr=lr, weight_decay=0.1, eps=1.5e-4)
            copy_states(optimizer, optimizer_aux)
            optimizer = torch.optim.AdamW(chain(params_wm, params_ac),
                                lr=lr, weight_decay=0.1, eps=1.5e-4)
            copy_states(optimizer_aux, optimizer)
        
        
        
        step+=1
        
        log_t = done_flag.astype(float).nonzero()[0]
        
        if len(log_t)>0:
            for log in log_t:
                scores.append(eps_reward[log].clone())
                
                recent_mean = 0
                if len(scores) > 0:
                     recent_scores = [s.item() for s in scores[-10:]] 
                     recent_mean = np.mean(recent_scores)
                
                current_eps = linearly_decaying_epsilon(2001, len_memory, 2000, 0)
                
                progress_bar.set_postfix({
                    'ep': len(scores),
                    'last': f'{eps_reward[log].item():.0f}',
                    'avg10': f'{recent_mean:.1f}',
                    'eps': f'{current_eps:.2f}'
                }, refresh=True)
                
            eps_reward[log_t]=0

save_checkpoint(model, model_target, optimizer, step, grad_step, f'checkpoints/{env_name}.pth')

def eval_phase(eval_runs=50, max_eval_steps=27000, num_envs=1):
    progress_bar = tqdm.tqdm(total=eval_runs)
    
    scores=[]
    
    state, info = env.reset()
    state = preprocess(state)
    print(f"init state {state.shape}")
    
    states = deque(maxlen=4)
    for i in range(4):
        states.append(state)
    
    
    eps_reward=torch.tensor([0]*num_envs, dtype=torch.float)
    
    reward=np.array([0]*num_envs)
    terminated=np.array([False]*num_envs)
    
    last_lives=np.array([0]*num_envs)
    life_loss=np.array([0]*num_envs)
    resetted=np.array([0])

    finished_envs=np.array([False]*num_envs)
    done_flag=0
    last_grad_update=0
    eval_run=0
    step=np.array([0]*num_envs)
    while eval_run<eval_runs:
        env.seed=SEED+eval_run
        model_target.train()
        
        Q_action = model_target.env_step(torch.cat(list(states),-3).unsqueeze(0).unsqueeze(0))
        action = epsilon_greedy(Q_action.squeeze(), 5000, 0.0005, num_envs).cpu()
        
        state, reward, terminated, truncated, info = env.step([action.numpy()] if num_envs==1 else action.numpy())
        state = preprocess(state)
        states.append(state)
        
        eps_reward+=reward

        
        done_flag = np.logical_or(terminated, truncated)
        lives = info['lives']
        life_loss = (last_lives-lives).clip(min=0)
        resetted = (lives-last_lives).clip(min=0)
        last_lives = lives        
        
        step+=1
        
        log_t = done_flag.astype(float).nonzero()[0]
        if len(log_t)>0:
            progress_bar.update(1)
            for log in log_t:
                if finished_envs[log]==False:
                    scores.append(eps_reward[log].clone())
                    eval_run+=1
                step[log]=0
                
            eps_reward[log_t]=0            
            for i, log in enumerate(step>max_eval_steps):
                if log==True and finished_envs[i]==False:
                    scores.append(eps_reward[i].clone())
                    step[i]=0
                    eval_run+=1
                    eps_reward[i]=0
            
    return scores



def eval(eval_runs=50, max_eval_steps=27000, num_envs=1):
    assert num_envs==1, 'The code for num eval envs > 1 is messed up.'
    
    scores = eval_phase(eval_runs, max_eval_steps, num_envs)    
    scores = torch.stack(scores)
    scores, _ = scores.sort()
    
    _25th = eval_runs//4

    iq = scores[_25th:-_25th]
    iqm = iq.mean()
    iqs = iq.std()

    print(f"Scores Mean {scores.mean()}")
    print(f"Inter Quantile Mean {iqm}")
    print(f"Inter Quantile STD {iqs}")

    
    plt.xlabel('Episode (Sorted by Reward)')
    plt.ylabel('Reward')
    plt.plot(scores)
    
    new_row = {'env_name': env_name, 'mean': scores.mean().item(), 'iqm': iqm.item(), 'std': iqs.item(), 'seed': SEED}
    add_to_csv('results.csv', new_row)

    with open(f'results/{env_name}-{SEED}.txt', 'w') as f:
        f.write(f" Scores Mean {scores.mean()}\n Inter Quantile Mean {iqm}\n Inter Quantile STD {iqs}")
    
    
    return scores

scores = eval(eval_runs=100, num_envs=1)