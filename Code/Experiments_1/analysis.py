#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/06/21 22:27:16
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   main
'''

# %% Import Package
# Basic
import os

# Add path
os.chdir(os.path.dirname(__file__))

# Self-defined
import utils
import recorder as rd

# %% Main Function
if __name__ == "__main__":
    analyst = rd.analysis.Analyst('../../Results')
    experiment_name = {'d': '08-12_18-07-39'}
    experiments = analyst.choose_experiments(experiment_name)
    experiments_results = analyst.read_json(experiments)
    analyst.draw_results(experiments_results, 
                    cv = [0,1,2,3,4],
                    metric_list=[[[{"train":"t_learner_train_loss"}, {'test': "t_learner_test_loss"}], {"test":"s_learner_test_loss"}]], 
                    model='merge')
            
# %%
