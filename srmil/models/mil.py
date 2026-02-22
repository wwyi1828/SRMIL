import torch
import torch.nn as nn
from torch.nn import functional as F

class BagNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, learnable=True, momentum=0.1):
        super(BagNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        if learnable:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('gamma', torch.zeros(num_features))
            self.register_buffer('beta', torch.ones(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        else:
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        with torch.no_grad():
            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum

        # x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        out = self.gamma * x_normalized + self.beta

        return out



class Attention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, output_dim=1, act='relu', bias=False, dropout=0, tau=1):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = output_dim # if >1, then class-aware attention

        self.tau = tau
        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        if act == 'gelu':
            self.attention.append(nn.GELU())
        if act == 'relu':
            self.attention.append(nn.ReLU())
        if act == 'tanh':
            self.attention.append(nn.Tanh())
        
        # self.attention += [nn.Dropout(dropout)]
        self.attention.append(nn.Linear(self.D, self.K, bias=bias))

        self.attention = nn.Sequential(*self.attention)
        # self.apply(initialize_weights)

    def forward(self, x, no_norm=False, hard=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2) # K x batch_size
        A_ori = A.clone()

        A = F.softmax(A/self.tau, dim=-1)
        if hard:
            max_index = A.max(dim=-1, keepdim=True)[1]
            hard_assigns = torch.zeros_like(A, memory_format=torch.legacy_contiguous_format).scatter_(-1, max_index, 1.0)
            A = hard_assigns - A.detach() + A
        # x, batch_size x input_dim
        x = torch.matmul(A, x) # --> K x input_dim

        if no_norm:
            return x, A_ori #
        else:
            return x, A