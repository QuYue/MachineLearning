#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   parameter.py
@Time    :   2022/08/08 13:27:19
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   parameter
'''

# %% Import Packages
# Basic
import os
import sys
import json
import random
import torch
import numpy as np
import datetime

# Add path
# os.chdir(os.path.dirname(__file__))

# Self-defined
if __package__ is None:
    sys.path.append('..')
    import record
else:
    from . import record

# Modules
import modeler as ml
from utils import tools

# %% Classes
class PARAM():
    def __init__(self) -> None:
        pass
    def demo__init__(self) -> None:
        # Random seed
        self.seed = 1       # Random seed
        # Device
        self.gpu = 0        # Used GPU, when bool (if use GPU), when int (the ID of GPU) 
        # Dataset
        self.dataset_name = "Synthetic"     # Dataset name
        self.train_valid_test = [4, 1, 1]   # The ratio of training, validation and test data
        self.cv = 5                         # Fold number for cross-validation
        # Dataset Parameters
        self.dataset_set = {"synthetic":
                                {"name": "Synthetic",
                                 "data_number": 10000,
                                 "data_dimensions": 10,
                                 "ifprint": False,
                                 "stratify": 't',
                                 "keylist": ['x', 't', 'y', 'potential_y'],
                                 "type_list": ['float', 'long', 'float', 'float']}}
        # Model
        self.model_name_list = ["s_learner", "t_learner"]   # Model name list
        # Model Parameters
        self.model_param_set = {"s_learner":
                                    {"name": "S_Learner", 
                                     "input_size": self.dataset_set[self.dataset_name.lower().strip()]["data_dimensions"],
                                     "output_size": 1,
                                     "hidden_size": 15,
                                     "layer_number": 3},
                                "t_learner":
                                    {"name": "T_Learner",
                                     "input_size":self.dataset_set[self.dataset_name.lower().strip()]["data_dimensions"],
                                     "output_size":1,
                                     "hidden_size":15,
                                     "layer_number":3}}
        # Training
        self.epochs = 10            # Epochs
        self.batch_size = 1000      # Batch size
        self.learn_rate = 0.01      # Learning rate
        self.test_epoch = 1         # Test once every few epochs
        # Records
        self.ifrecord = True                # If record
        self.now = datetime.datetime.now()  # Current time

        # Setting
        self.setting()

    def setting(self):
        self.random_setting()
        self.device_setting(True)
        self.dataset_setting()
        self.model_setting()
    
    @property
    def train_ratio(self):
        return self.train_valid_test[0] / (self.train_valid_test[0]+self.train_valid_test[2])
    
    def random_setting(self):
        # Setting of random seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)   # If you are using multi-GPU.

    def model_setting(self):
        # Setting of models
        self.model_param_list = [self.model_param_setting(name) for name in self.model_name_list]
        self.model_list = [ml.get_model(param.name, param.dict) for param in self.model_param_list]
    
    def model_device_setting(self, device=None):
        # Setting of models and devices
        if device is None:
            device = self.device
        self.model_list = [model.to(device) for model in self.model_list]

    def model_param_setting(self, model_name):
        # Setting of parameters of models
        model_param = tools.MyStruct('model_param', [model_name])
        name = model_name.lower().strip()
        if name in self.model_param_set:
            model_param.__dict__.update(self.model_param_set[name])
        else:
            raise ValueError("Model name is not defined.")
        return model_param
    
    def dataset_setting(self):
        # Setting of dataset
        self.dataset = tools.MyStruct('dataset', [self.dataset_name])
        self.dataset.cv = self.cv
        self.dataset.train_ratio = self.train_ratio
        name = self.dataset_name.lower().strip()
        if name in self.dataset_set:
            self.dataset.__dict__.update(self.dataset_set[name])
        else:
            raise ValueError("Dataset name is not existed.")

    def device_setting(self, ifprint=True):
        # Setting of device
        if isinstance(self.gpu, bool):
            if self.gpu:
                try:
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0")
                except:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        elif isinstance(self.gpu, int):
            device_count = max(torch.cuda.device_count(), 1)
            self.gpu = min(self.gpu, device_count-1)
            if self.gpu >= 0:
                try:
                    self.device = torch.device("cuda:" + str(self.gpu) if torch.cuda.is_available() else "mps:" + str(self.gpu))
                except:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            raise ValueError("The type of gpu must be bool or int.")
        if self.device == torch.device("cpu"):
            device = "CPU"
        else:
            try:
                device = f"GPU {self.device.index} ({torch.cuda.get_device_name(self.device.index)})"
            except:
                device = f"GPU {self.device.index} ({self.device.type})"
        self.device_name = device
        if ifprint:
            print(f"The experimental environment is set to {device}.")

    def tojson(self):
        # Convert to json
        param_dict = self.__dict__.copy()
        param_dict.pop('device')
        param_dict.pop('dataset')
        param_dict.pop('model_param_list')
        param_dict.pop('model_list')
        param_dict['now'] = param_dict['now'].strftime("%Y-%m-%d %H:%M:%S")
        param_dict['recorder'] = self.recorder.copy()
        for key in param_dict["recorder"].keys():
            param_dict['recorder'][key] = param_dict['recorder'][key].tojson()
        return param_dict
    
    def save(self, filename=None):
        info = self.tojson()
        if filename is None:
            filename = f"{self.now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        filepath = os.path.join(self.save_path, filename)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(info, f)

    def load_json(self, json):
        # Load from json
        self.__dict__.update(json)
        self.random_setting()
        self.dataset_setting()
        for k, v in self.recorder.items():
            temp = record.Recorder()
            temp.load_json(v)
            self.recorder[k] = temp

# %% Functions
def read_json(path):
    Parm = PARAM()
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    Parm.load_json(d)
    return Parm

def main():
    pass

# %% Main Function
if __name__ == '__main__':
    main()