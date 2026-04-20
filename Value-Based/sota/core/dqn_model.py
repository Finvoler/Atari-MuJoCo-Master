import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.resnet import IMPALA_Resnet, DQN_Conv
from networks.mlp import MLP
from networks.weight_init import init_xavier
from utils.general import params_count

def renormalize(tensor, has_batch=False):
    shape = tensor.shape
    tensor = tensor.view(tensor.shape[0], -1)
    max_value,_ = torch.max(tensor, -1, keepdim=True)
    min_value,_ = torch.min(tensor, -1, keepdim=True)
    return ((tensor - min_value) / (max_value - min_value + 1e-5)).view(shape)


class DQN(nn.Module):
    def __init__(self, n_actions, hiddens=2048, mlp_layers=1, scale_width=4,
                 n_atoms=51, Vmin=-10, Vmax=10, num_buckets=51):
        super().__init__()
        self.support = torch.linspace(Vmin, Vmax, n_atoms).cuda()
        self.num_buckets = num_buckets
        self.n_actions = n_actions
        self.batch = 0

        self.hiddens=hiddens
        self.scale_width=scale_width
        self.act = nn.ReLU()


        self.encoder_cnn = IMPALA_Resnet(scale_width=scale_width, norm=False, init=init_xavier, act=self.act)




        self.projection = MLP(13824, med_hiddens=hiddens, out_hiddens=hiddens,
                              last_init=init_xavier, layers=1)
        self.prediction = MLP(hiddens, out_hiddens=hiddens, layers=1, last_init=init_xavier)

        self.transition = nn.Sequential(DQN_Conv(32*scale_width+n_actions, 32*scale_width, 3, 1, 1, norm=False, init=init_xavier, act=self.act),
                                        DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=False, init=init_xavier, act=self.act))




        self.a = MLP(hiddens, out_hiddens=n_actions*num_buckets, layers=1, in_act=self.act, last_init=init_xavier)
        self.v = MLP(hiddens, out_hiddens=num_buckets, layers=1, in_act=self.act, last_init=init_xavier)

        params_count(self, 'DQN')

    def forward(self, X, y_action):
        X, z = self.encode(X)


        q, action = self.q_head(X)
        z_pred = self.get_transition(z, y_action)

        return q, action, X[:,1:].clone().detach(), z_pred


    def env_step(self, X):
        with torch.no_grad():
            X, _ = self.encode(X)
            _, action = self.q_head(X)

            return action.detach()


    def encode(self, X):
        batch, seq = X.shape[:2]
        self.batch = batch
        self.seq = seq
        X = self.encoder_cnn(X.contiguous().view(self.batch*self.seq, *(X.shape[2:])))
        X = renormalize(X).contiguous().view(self.batch, self.seq, *X.shape[-3:])
        X = X.contiguous().view(self.batch, self.seq, *X.shape[-3:])
        z = X.clone()
        X = X.flatten(-3,-1)
        X = self.projection(X)
        return X, z

    def get_transition(self, z, action):
        z = z.contiguous().view(-1, *z.shape[-3:])


        num_actions = action.shape[1] if len(action.shape) > 1 else 1

        action = F.one_hot(action.clone(), self.n_actions).view(-1, self.n_actions)
        action = action.view(-1, num_actions, self.n_actions, 1, 1).expand(-1, num_actions, self.n_actions, *z.shape[-2:])

        z_pred = torch.cat( (z, action[:,0]), 1)
        z_pred = self.transition(z_pred)
        z_pred = renormalize(z_pred)

        z_preds=[z_pred.clone()]


        for k in range(num_actions - 1):
            z_pred = torch.cat( (z_pred, action[:,k+1]), 1)
            z_pred = self.transition(z_pred)
            z_pred = renormalize(z_pred)

            z_preds.append(z_pred)


        z_pred = torch.stack(z_preds,1)

        z_pred = self.projection(z_pred.flatten(-3,-1)).view(self.batch, num_actions, -1)
        z_pred = self.prediction(z_pred)

        return z_pred


    def q_head(self, X):
        q = self.dueling_dqn(X)
        action = (q*self.support).sum(-1).argmax(-1)

        return q, action

    def get_max_action(self, X):
        with torch.no_grad():
            X, _ = self.encode(X)
            q = self.dueling_dqn(X)

            action = (q*self.support).sum(-1).argmax(-1)
            return action

    def evaluate(self, X, action):
        with torch.no_grad():
            X, _ = self.encode(X)

            q = self.dueling_dqn(X)

            action = action[:,:,None,None].expand_as(q)[:,:,0][:,:,None]
            q = q.gather(-2,action)

            return q

    def dueling_dqn(self, X):
        X = F.relu(X)

        a = self.a(X).view(self.batch, -1, self.n_actions, self.num_buckets)
        v = self.v(X).view(self.batch, -1, 1, self.num_buckets)

        q = v + a - a.mean(-2,keepdim=True)
        q = F.softmax(q,-1)

        return q

    def network_ema(self, rand_network, target_network, alpha=0.5):
        for param, param_target in zip(rand_network.parameters(), target_network.parameters()):
            param_target.data = alpha * param_target.data + (1 - alpha) * param.data.clone()

    def hard_reset(self, random_model, alpha=0.5):
        with torch.no_grad():

            self.network_ema(random_model.encoder_cnn, self.encoder_cnn, alpha)
            self.network_ema(random_model.transition, self.transition, alpha)

            self.network_ema(random_model.projection, self.projection, 0)
            self.network_ema(random_model.prediction, self.prediction, 0)

            self.network_ema(random_model.a, self.a, 0)
            self.network_ema(random_model.v, self.v, 0)
