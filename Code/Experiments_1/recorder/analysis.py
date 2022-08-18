#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   analysis.py
@Time    :   2022/08/12 14:14:48
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   analysis
'''

# %% Import Packages
# Basic
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# Add path
# os.chdir(os.path.dirname(__file__))

# Self-defined
if __package__ is None:
    sys.path.append('..')
    import parameter
else:
    from . import parameter

# Modules
parameter.PARAM()

# %% Classes
class Analyst():
    def __init__(self, result_path="../../../Results") -> None:
        # self.data_name = 'IHDP'
        self.experiment_name = {'a':'08-12_13-46-58'}
        self.result_path = result_path

    def get_experiments(self):
        listdir = os.listdir(self.result_path) 
        self.fold_list = []
        for fold_name in listdir:
            if 'experiment' in fold_name.lower():
                fold_path = os.path.join(self.result_path, fold_name)
                if os.path.isdir(fold_path):
                    self.fold_list.append(fold_path)

        self.experiment_total_list = []
        for fold_name in self.fold_list:
            if os.path.exists(fold_name):
                listdir = os.listdir(fold_name) 
                for experiment_name in listdir:
                    p = os.path.join(fold_name, experiment_name)
                    self.experiment_total_list.append(p)

    def choose_experiments(self, experiment_name=None):
        if experiment_name is None:
            experiment_name = self.experiment_name
        self.get_experiments()
        experiments = dict()
        for e_name, e_id in experiment_name.items():
            if isinstance(e_id, list):
                for e in e_id:
                    for path in self.experiment_total_list:
                        if e in path:
                            json_file = os.path.join(path, 'results.json')
                            if os.path.isfile(json_file):
                                if e_name not in experiments:
                                    experiments[e_name] = []
                                experiments[e_name].append(json_file)
                                break
            else:
                for path in self.experiment_total_list:
                    if e_id in path:
                        json_file = os.path.join(path, 'results.json')
                        if os.path.isfile(json_file):
                            experiments[e_name] = json_file
                        break
        return experiments

    def read_json(self, experiments):
        experiments_results = dict()
        for e_name, e_path in experiments.items():
            if isinstance(e_path, list):
                for path in e_path:
                    data = parameter.read_json(path)
                    if e_name not in experiments_results:
                        experiments_results[e_name] = [data]
                    else:
                        experiments_results[e_name].append(data)
            else:
                data = parameter.read_json(e_path)
                experiments_results[e_name] = data
        return experiments_results

    def _draw(self, metric, ax, cv, experiments_results, model):
        if isinstance(metric, list):
            key_list = []
            value_list = []
            for m in metric:
                key = list(m.keys())[0]
                value = m[key]
                key_list.append(key)
                value_list.append(value)

                for e_name, e_data in experiments_results.items():
                    experiment_record = e_data.recorder[key]
                    epoch = experiment_record.index
                    data = experiment_record.query(value)

                    epoch = epoch[0]
                    data = [data[c] for c in cv]
                    name = [f"{e_name}_{c}" for c in cv]

                    if model == 'merge':
                        data = np.array(data)
                        data_mean = np.mean(data, axis=0)
                        data_std = np.std(data, axis=0)
                        ax.plot(epoch, data_mean, label=f"{e_name}|{key}:{value}")
                        ax.fill_between(epoch, data_mean-data_std, data_mean+data_std, alpha=0.2)
                    elif model == 'split':
                        for i in range(len(data)):
                            ax.plot(epoch, data[i], label=f"{name[i]}|{key}:{value}")
            
            key_flag = True
            for i in range(len(key_list)-1):
                if key_list[0] != key_list[i+1]:
                    key_flag = False
                    break
            
            value_flag = True
            for i in range(len(value_list)-1):
                if value_list[0] != value_list[i+1]:
                    value_flag = False
                    break
            
            text=''
            if key_flag and value_flag:
                text = f"{key_list[0]}:{value_list[0]}"
            elif key_flag:
                text = key_list[0]
            elif value_flag:
                text = value_list[0]
            else:
                text = 'Compare'
                
            ax.title.set_text(text)
            ax.legend()
            ax.xaxis.grid(True)
            ax.yaxis.grid(True) 
            ax.set_xlabel('Epoch')
            ax.set_ylabel(text)
        else:
            key = list(metric.keys())[0]
            value = metric[key]

            for e_name, e_data in experiments_results.items():
                experiment_record = e_data.recorder[key]
                epoch = experiment_record.index
                data = experiment_record.query(value)

                epoch = epoch[0]
                data = [data[c] for c in cv]
                name = [f"{e_name}_{c}" for c in cv]

                if model == 'merge':
                    data = np.array(data)
                    data_mean = np.mean(data, axis=0)
                    data_std = np.std(data, axis=0)
                    ax.plot(epoch, data_mean, label=e_name)
                    ax.fill_between(epoch, data_mean-data_std, data_mean+data_std, alpha=0.2)
                elif model == 'split':
                    for i in range(len(data)):
                        ax.plot(epoch, data[i], label=name[i])
                ax.title.set_text(f"{key}:{value}")
                ax.legend()
                ax.xaxis.grid(True)
                ax.yaxis.grid(True) 
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f"{key}:{value}")
                
    def draw_results(self, experiments_results, metric_list=[{'train':'loss'}], cv=[0,1,2,3,4], model='merge'):
        for metric in metric_list:
            fig = plt.figure()
            if isinstance(metric, list):
                for s, m in enumerate(metric):
                    ax = fig.add_subplot(1, len(metric), s+1)
                    self._draw(m, ax, cv, experiments_results, model)
            else:
                ax = fig.add_subplot(111)
                self._draw(metric, ax, cv, experiments_results, model)
            # plt.savefig(f"{metric}.png")
            plt.show()
            
# %% Functions
def main():
    pass

# %% Main Function
if __name__ == '__main__':
    main()


