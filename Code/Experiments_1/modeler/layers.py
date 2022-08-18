#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   layer.py
@Time    :   2022/07/06 07:34:08
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   layers
'''

# %% Import Packages
# Basic
import torch
import torch.nn as nn

# %% Fully Connected Layer (Classes)
class FullyConnected(nn.Module):
    """
    Fully Connected Layer.
    """
    def __init__(self, in_dim, out_dim, dropout=0, other_layers=None):
        super(FullyConnected, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.FullyConnected = nn.Sequential()
        self.FullyConnected.add_module('Linear', nn.Linear(self.in_dim, self.out_dim))
        if self.dropout != 0:
            self.FullyConnected.add_module('Dropout', nn.Dropout(self.dropout))
        if other_layers is not None:
            for layer in other_layers:
                self.FullyConnected.add_module(layer.__class__.__name__, layer)

    def forward(self, tensors:torch.Tensor):
        return self.FullyConnected(tensors)


# %% Main Function
if __name__ == '__main__':
    pass