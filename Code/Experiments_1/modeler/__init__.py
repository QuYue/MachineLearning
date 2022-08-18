#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2022/06/23 01:11:14
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   __init__ of modeler
'''

# %% Import Packages
# Basic

# Self-defined

# Modules
from .s_learner import S_Learner
from .t_learner import T_Learner

# %% Functions
def get_model(model_name, kwargs):
    """
    Get model.
    """
    if model_name == "S_Learner":
        return S_Learner(**kwargs)
    elif model_name == "T_Learner":
        return T_Learner(**kwargs)
    else:
        raise ValueError(f"There is no model called '{model_name}'.")

def train(model_list):
    if isinstance(model_list, list):
        for model in model_list:
            model.train()
    else:
        model_list.train()

def eval(model_list):
    if isinstance(model_list, list):
        for model in model_list:
            model.eval()
    else:
        model_list.eval()

# %% Main Function
if __name__ == '__main__':
    pass