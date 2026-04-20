import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.weight_init import *
from utils.general import params_count, nsd_Module

import numpy as np

class DQN_Conv(nn.Module):
    def __init__(self, in_hiddens, hiddens, ks, stride, padding=0, max_pool=False, norm=True, init=init_relu, act=nn.SiLU(), bias=True):
        super().__init__()

        self.conv = nn.Sequential(
                                  nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, bias=bias),
                                  nn.MaxPool2d(3,2,padding=1) if max_pool else nn.Identity(),
                                  (nn.GroupNorm(32, hiddens, eps=1e-6) if hiddens%32==0 else nn.BatchNorm2d(hiddens, eps=1e-6)) if norm else nn.Identity(),
                                  act,
                                  )
        self.conv.apply(init)

    def forward(self, X):
        return self.conv(X)

class DQN_CNN(nn.Module):
    def __init__(self, in_hiddens, hiddens, ks, stride, padding=0):
        super().__init__()

        self.cnn = nn.Sequential(DQN_Conv(4, 32, 8, 4),
                                 DQN_Conv(32, 64, 4, 2),
                                 DQN_Conv(64, 64, 3, 1)
                                )
    def forward(self, X):

        return self.cnn(X)

class Residual_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, act=nn.SiLU(), out_act=nn.SiLU(), norm=True, init=init_xavier, bias=True):
        super().__init__()

        conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1,
                                            stride=stride, bias=bias),
                              (nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6)) if norm else nn.Identity(),
                              act)
        conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                              (nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6)) if norm else nn.Identity(),
                              out_act)

        conv1.apply(init)
        conv2.apply(init if out_act!=nn.Identity() else init_xavier)

        self.conv = nn.Sequential(conv1, conv2)

        self.proj=nn.Identity()
        if stride>1 or in_channels!=channels:
            self.proj = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1,
                        stride=stride)

        self.proj.apply(init_proj2d)
        self.out_act = out_act

    def forward(self, X):
        Y = self.conv(X)
        return Y+self.proj(X)

class IMPALA_YY(nsd_Module):
    def __init__(self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU()):
        super().__init__()

        self.yin = self.get_yin(1, 16*scale_width, 32*scale_width)

        self.yang = self.get_yang(first_channels, 16*scale_width)

        self.head = nn.Sequential(self.get_yang(16*scale_width, 32*scale_width),
                                  self.get_yang(32*scale_width, 32*scale_width, last_relu=True))

        params_count(self, 'IMPALA ResNet')

    def get_yin(self, in_hiddens, hiddens, out_hiddens):
        blocks = nn.Sequential(DQN_Conv(in_hiddens, hiddens, 3, 1, 1, max_pool=True, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(hiddens, hiddens, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(hiddens, hiddens, norm=self.norm, act=self.act, init=self.init),



                              )
        return blocks

    def get_yang(self, in_hiddens, out_hiddens, last_relu=False):

        blocks = nn.Sequential(DQN_Conv(in_hiddens, out_hiddens, 3, 1, 1, max_pool=True, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init, out_act=self.act if last_relu else nn.Identity())
                              )

        return blocks

    def forward(self, X):

        y = self.yin(X[:,-3:].mean(-3)[:,None])
        x = self.yang(X)


        X = 0.67*x + 0.33*y

        return self.head(X)


class IMPALA_Resnet(nsd_Module):
    def __init__(self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU(), bias=True):
        super().__init__()

        self.cnn = nn.Sequential(self.get_block(first_channels, 16*scale_width),
                                 self.get_block(16*scale_width, 32*scale_width),
                                 self.get_block(32*scale_width, 32*scale_width, last_relu=True))
        params_count(self, 'IMPALA ResNet')

    def get_block(self, in_hiddens, out_hiddens, last_relu=False):

        blocks = nn.Sequential(DQN_Conv(in_hiddens, out_hiddens, 3, 1, 1, max_pool=True, bias=self.bias, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init, out_act=self.act if last_relu else nn.Identity())
                              )

        return blocks

    def forward(self, X):
        return self.cnn(X)
