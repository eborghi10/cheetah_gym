import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=1024, hidden2=1024, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(hidden1, 256)
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, hidden2)
        self.fc10 = nn.Linear(hidden2, nb_actions)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.7)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        # out1 = self.dropout(out)
        out = self.fc2(out)
        out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc3(out)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc4(out)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc5(out)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc6(out)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc7(out + out1)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc8(out)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = self.fc9(out)
        # out = self.gelu(out)
        # out = self.dropout(out)
        out = self.fc10(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=1024, hidden2=1024, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.gelu = nn.GELU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.gelu(out)
        out = self.fc3(out)
        return out