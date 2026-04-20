import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.general import nsd_Module
from networks.weight_init import *

class MLP(nn.Module):
    def __init__(self, in_hiddens=512, med_hiddens=512, out_hiddens=512, layers=1,
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_xavier, bias=True):
        super().__init__()


        modules=[]
        self.init=init
        self.last_init=last_init

        hiddens=in_hiddens
        _out_hiddens = med_hiddens
        act = in_act
        for l in range(layers):
            last_layer = l==(layers-1)
            if last_layer:
                _out_hiddens = out_hiddens
                act = out_act
            modules.append(nn.Linear(hiddens, _out_hiddens, bias=bias))

            modules.append(act)
            hiddens=med_hiddens
        self.mlp=nn.Sequential(*modules)



        self.init_weights()

    def turn_off_grads(self):
        for layer in self.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.requires_grad=False
            if hasattr(layer, 'bias'):
                layer.bias.requires_grad=False
    def init_weights(self):
        self.mlp.apply(self.init)
        self.mlp[-2].apply(self.last_init)


    def forward(self,X):
        return self.mlp(X)
