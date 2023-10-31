import torch
from torch import nn
from d2l import torch as d2l

class SpaceGroupPred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(SpaceGroupPred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)