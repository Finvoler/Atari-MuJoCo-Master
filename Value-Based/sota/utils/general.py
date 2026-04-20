import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import os
import math
import inspect
import pandas as pd
import einops


class Hypers:
    def __init__(self, max_depth=3, **kwargs):
        super().__init__(**kwargs)
        self.save_hypers(max_depth)

    def save_hypers(self, max_depth, ignore=[]):
      """Save function arguments into class attributes."""
      seen_init=False
      frame = inspect.currentframe()
      for d in range(max_depth):
          frame = frame.f_back
          if frame.f_back and frame.f_back.f_code.co_name == "__init__":
              seen_init=True
          if seen_init and frame.f_back.f_code.co_name != "__init__":
              break
      _, _, _, local_vars = inspect.getargvalues(frame)
      self.hparams = {k:v for k, v in local_vars.items()
          if k not in set(ignore+['self']) and not k.startswith('_')}
      for k, v in self.hparams.items():
          setattr(self, k, v)

class nsd_Module(Hypers, nn.Module):
    def __init__(self):
        super().__init__(max_depth=3)


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super(Rearrange, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return einops.rearrange(x, self.pattern)


def add_to_csv(path, new_row):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=new_row.keys())

    df.loc[len(df.index)] = new_row
    df.to_csv(path, index=False)


def params_count(model, name='Model'):
    params_to_count = [p for p in model.parameters() if p.requires_grad]
    parameters_count = sum(p.numel() for p in params_to_count)
    print(f'{name} Parameters: {parameters_count/1e6:.2f}M')
    return parameters_count

def params_and_grad_norm(model):
    param_norm, grad_norm = 0, 0
    for n, param in model.named_parameters():
        if not n.endswith('.bias'):
            param_norm += torch.norm(param.data)
            if param.grad is not None:
                grad_norm += torch.norm(param.grad)
    return param_norm, grad_norm

def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
