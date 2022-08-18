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
import torch
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

# Add path
os.chdir(os.path.dirname(__file__))

# Self-defined
import utils
import dataprocessor as dp
import modeler as ml
import recorder as rd

# %% Set Super-parameters
class MyParam(rd.parameter.PARAM):
    def __init__(self) -> None:
        # Random seed
        self.seed = 1       # Random seed
        # Device
        self.gpu = -1        # Used GPU, when bool (if use GPU), when int (the ID of GPU) 
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
                                 "typelist": ['float', 'long', 'float', 'float']}}
        # Model
        self.model_name_list = ["s_learner", "t_learner"]   # Model name list
        self.model_save = True                 # Whether to save the model
        # Model Parameters
        self.model_param_set = {"s_learner":
                                    {"name": "S_Learner", 
                                     "input_size": self.dataset_set[self.dataset_name.lower().strip()]["data_dimensions"],
                                     "output_size": 1,
                                     "hidden_size": 15,
                                     "layer_number": 5,
                                     "dropout": 0,
                                     "optimizer": {"name": "SGD", "lr": 0.001}},
                                "t_learner":
                                    {"name": "T_Learner",
                                     "input_size":self.dataset_set[self.dataset_name.lower().strip()]["data_dimensions"],
                                     "output_size":1,
                                     "hidden_size":15,
                                     "layer_number":5,
                                     "dropout": 0,
                                     "optimizer": {"name": "SGD", "lr": 0.001}}}
        # Training
        self.epochs = 10            # Epochs
        self.batch_size = 1000      # Batch size
        self.learn_rate = 0.01      # Learning rate
        self.test_epoch = 1         # Test once every few epochs
        # Records
        self.ifrecord = True                # If record
        self.now = datetime.datetime.now()  # Current time
        self.recorder = dict()                # Recorder (initialization)
        self.save_path = f"../../Results/Experiments_1/{self.now.strftime('%Y-%m-%d_%H-%M-%S')}"
        # Checkpoints
        self.ifcheckpoint = True            # If checkpoint
        # Setting
        self.setting()

Parm = MyParam()

# %% Main Function
if __name__ == "__main__":
    print("Loading dataset ...")
    dataset = dp.datasets.load_dataset(Parm.dataset_name, seed=Parm.seed, **Parm.dataset.dict)
    print("Start training ...")
    Parm.recorder['train'] = rd.record.Recorder_nones([dataset.cv, Parm.epochs])
    Parm.recorder['test'] = rd.record.Recorder_nones([dataset.cv, Parm.epochs])
    for cv in range(dataset.cv):
        Parm.model_setting()        # Models initialization
        Parm.model_device_setting() # Models device setting
        print(f"Cross Validation {cv}: {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        train_loader, test_loader = dp.process.dataloader(dataset[cv], batch_size=Parm.batch_size, **Parm.dataset.dict)
        start_cv = time.time()
        for epoch in range(Parm.epochs):
            # Training
            train_record = rd.record.Record(index=epoch)
            ml.train(Parm.model_list) # Train model
            for batch_idx, data in enumerate(train_loader):
                data = [data.to(Parm.device) for data in data]
                data = dict(zip(Parm.dataset.keylist, data))
                batchrecord = rd.record.BatchRecord(size=data['x'].shape[0], index=batch_idx) 
                for model_name in Parm.model_name_list:
                    batchrecord[f"{model_name}_train_loss"] = []
                # Model
                pred_list = []
                for model_id, model in enumerate(Parm.model_list):
                    model_name = Parm.model_name_list[model_id]
                    loss = model.on_train(data)
                    batchrecord[f'{model_name}_train_loss'].append(loss.item() * data['x'].shape[0])
                train_record.add_batch(batchrecord)
            train_record['time'] = time.time()
            for model_name in Parm.model_name_list:
                train_record.aggregate({f'{model_name}_train_loss': 'mean_size'})

            # Testing
            test_record = rd.record.Record(index=epoch)
            ml.eval(Parm.model_list) # Testing model
            for batch_idx, data in enumerate(test_loader):
                data = [data.to(Parm.device) for data in data]
                data = dict(zip(Parm.dataset.keylist, data))
                batchrecord = rd.record.BatchRecord(size=data['x'].shape[0], index=batch_idx) 
                for model_name in Parm.model_name_list:
                    batchrecord[f"{model_name}_test_loss"] = []
                # Model
                for model_id, model in enumerate(Parm.model_list):
                    model_name = Parm.model_name_list[model_id]
                    loss = model.on_test(data)
                    batchrecord[f'{model_name}_test_loss'].append(loss * data['x'].shape[0])
                test_record.add_batch(batchrecord)
            test_record['time'] = time.time()
            for model_name in Parm.model_name_list:
                test_record.aggregate({f'{model_name}_test_loss': 'mean_size'})

            Parm.recorder['train'][cv, epoch] = train_record
            Parm.recorder['test'][cv, epoch] = test_record

            train_str = train_record.print_str([f'{model_name}_train_loss' for model_name in Parm.model_name_list])
            test_str = test_record.print_str([f'{model_name}_test_loss' for model_name in Parm.model_name_list])
            print(f"Epoch {epoch} | {test_str} time: {train_record['time'] - start_cv :.1f}s")
        # Parm.save(f"cv{cv}.json")
    
    print("Finish training ...\n")
    if Parm.ifrecord:
        Parm.save("results.json")
        print("Records saved.")
    print("Done!")

# %%
plt.figure(1)
for model_name in Parm.model_name_list:
    plt.plot(Parm.recorder['train'].query(f'{model_name}_train_loss')[0], label=f'{model_name}_train_loss')[0]
    plt.plot(Parm.recorder['test'].query(f'{model_name}_test_loss')[0], label=f'{model_name}_test_loss')[0]
plt.legend()
plt.grid('on')
plt.show()
# %%
