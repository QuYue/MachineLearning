#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   s_learner.py
@Time    :   2022/07/05 07:35:54
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   s_learner
'''

# %% Import Packages
# Basic
import torch
import torch.nn as nn

# Modules
if __package__ is None:
    import layers
else:
    from . import layers

# %% S_Learner (Classes)
class S_Learner(nn.Module):
    """
    S_Learner.
    """
    def __init__(self, input_size, output_size=1, hidden_size=10, layer_number=3, dropout=0.5, optimizer={"name": "Adam"}, **kwargs):
        """
        Initialize S_Learner model.
        """
        super(S_Learner, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_number = layer_number
        self.dropout = dropout
        self.optimizer_param = optimizer
        self.create_model()
        self.create_optimizer()
        self.create_loss_function()

    def create_model(self):
        """
        Create model.
        """
        self.net = nn.Sequential()
        self.net.add_module('fc0', nn.Linear(self.input_size+1, self.hidden_size))
        for i in range(self.layer_number-1):
            self.net.add_module(f'fc{i+1}', layers.FullyConnected(self.hidden_size, self.hidden_size, self.dropout, [nn.ReLU()]))
        self.net.add_module(f'fc_{self.layer_number}', layers.FullyConnected(self.hidden_size, self.output_size))

    def create_optimizer(self):
        """
        Create optimizer.
        """
        if self.optimizer_param["name"] == "Adam":
            optimizer = torch.optim.Adam
        elif self.optimizer_param["name"] == "SGD":
            optimizer = torch.optim.SGD
        else:
            raise ValueError(f"There is no optimizer called '{self.optimizer_param['name']}'.")
        optimizer_param = self.optimizer_param.copy()
        optimizer_param.pop("name")
        self.optimizer = optimizer(self.parameters(), **optimizer_param)

    def create_loss_function(self):
        self.MSELoss = nn.MSELoss()

    def forward(self, x):
        """
        Forward propagation.
        """
        sample_number = x.shape[0]
        pred_y0 = self.net(torch.cat([x, torch.zeros([sample_number, 1]).to(x.device)], dim=1))
        pred_y1 = self.net(torch.cat([x, torch.ones([sample_number, 1]).to(x.device)], dim=1))
        return pred_y0, pred_y1
    
    def predict(self, data):
        """
        Predict.
        """
        if isinstance(data, dict):
            x = data["x"]
        else:
            x = data
        pred = self.forward(x)
        pred = torch.cat(pred, dim=1)
        return {"y_pred": pred}

    def loss(self, data, pred):
        """
        Loss.
        """
        y = data["y"]
        t = data["t"]
        y_pred = pred["y_pred"]
        y_pred = y_pred[:, 0:1] * (1-t) + y_pred[:, 1:2] * t
        loss = self.MSELoss(y_pred, y)
        return loss

    def on_train(self, data):
        """
        Training.
        """
        self.train()
        pred = self.predict(data)
        self.optimizer.zero_grad()
        loss = self.loss(data, pred)
        loss.backward()
        self.optimizer.step()
        return loss

    def on_test(self, data):
        """
        Testing.
        """
        self.eval()
        pred = self.predict(data)
        loss = self.loss(data, pred).item()
        return loss


# %% Main Function
if __name__ == '__main__':
    x = torch.ones([10, 5])
    t = torch.ones([10, 1])

    model = S_Learner(5, 1, 10, 3)
    pred_y0, pred_y1 = model(x)
    print(f"x: {x.shape}")
    print(f"t: {t.shape}")
    print(f"pred_y0: {pred_y0.shape}")
    print(f"pred_y1: {pred_y1.shape}")