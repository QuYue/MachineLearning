#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   process.py
@Time    :   2022/07/06 09:55:49
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   process
'''

# %% Import Packages
# Basic
import os
import sys
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# Add path
if __package__ is None:
    os.chdir(os.path.dirname(__file__))
    sys.path.append("..")

# Self-defined
import utils

# %% Classes
# One fold dataset for cross-validation
class OneFoldDataSet(utils.tools.MyStruct):
    """
    A class for one fold DataSet (class).
    """
    def __init__(self, fold_name='Fold0'):
        super().__init__('OneFoldDataSet', [fold_name])

    @property
    def set(self):
        return list(self.dict)

    def keys(self, setname='train'):
        k = eval(f"list(self.{setname}.keys())")
        return k


# Dataset 
class DataSet(utils.tools.MyStruct):
    """
    A class for DataSet (class).
    """
    def __init__(self, dataset_name='dataset'):
        super().__init__('DataSet', [dataset_name])
        self.data = []

    def __len__(self):
        return len(self.data)
    
    @property
    def cv(self):
        return len(self.data)

    def tolist(self):
        return self.data

    def append(self, value):
        self.data.append(value)
    
    def pop(self, index):
        return self.data.pop(index)

    def __getitem__(self, *inputs):
        if isinstance(inputs[0], tuple):
            inputs = inputs[0]
            return [self.data[i] for i in inputs]
        elif isinstance(inputs[0], int):
            return self.data[inputs[0]]
        elif isinstance(inputs[0], slice):
            return self.data[inputs[0]]
        else:
            raise Exception('Input type must be int or slice')

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if len(self.data) == key:
                self.data.append(value)
            elif len(self.data) > key:
                self.data[key] = value
            self.data[key] = value
        elif isinstance(key, tuple):
            if len(key) != len(value):
                raise Exception('Input length must be equal')
            for i in range(len(key)):
                self.__setitem__(key[i], value[i])
        elif isinstance(key, slice):
            thelist = list(range(len(self.data)))
            theindex = thelist[key]
            if len(theindex) != len(value):
                raise Exception('Input length must be equal')
            for i in theindex:
                self.__setitem__(theindex[i], value[i])
    
    def __repr__(self):
        """
        Print
        """
        return self.__struct_name__ + self.content_print + f" with {len(self.data)}-folds" # put constructor arguments in the ()


# %% Functions
# Data split
def data_split(data, dataset_name, train_ratio=0.8, cv=1, seed=1, stratify=None, **kwargs):
    '''
    The function of splitting data which can: 
        - Split the dataset into train/test split.
        - Also split the dataset proportionally by stratify.
        - Also split the dataset into train/test split by cross-validation.
    
    Parameters
    ----------
    data: dict
        The data.
    dataset_name: str
        Dataset name.
    train_ratio: float, optional
        The ratio of trainset. (default: 0.8)
    cv: int, optional
        The number of folds for cross-validation. (default: 1)
    seed: int, optional
        Random seed. (default: None)
    kwargs: dict, keyword arguments
        Other parameters. (default: {})

    Returns
    -------
    dataset: DataSet
        The dataset.
    '''
    # If data is dict
    if not isinstance(data, dict):
        raise ValueError("The type of data must be dict.")
    # Initial dataset
    dataset = []
    # Split dataset
    keys = list(data.keys())
    values = list(data.values())
    # Stratify
    stratify = data[stratify] if stratify in data else None
    # Dataset initialize
    dataset = DataSet(dataset_name)

    if cv <= 1:
        # Split dataset
        train_test_data = train_test_split(*values, train_size=train_ratio, random_state=seed, stratify=stratify)
        i, d = 0, OneFoldDataSet(f'Fold{0}')
        d.train = dict()
        d.test = dict()
        for key in keys:
            d.train[key] = train_test_data[i]
            d.test[key] = train_test_data[i+1]
            i += 2
        dataset.append(d)
    else:
        if stratify is None:
            stratify = data[keys[0]]
            # KFold
            kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        else:
            # StratifiedKFold
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        for f, (train_index, test_index) in enumerate(kf.split(stratify, stratify)):
            d = OneFoldDataSet(f'Fold{f}')
            d.train = dict()
            d.test = dict()
            for i, key in enumerate(keys):
                d.train[key] = values[i][train_index]
                d.test[key] = values[i][test_index]
            dataset.append(d)
    return dataset


# CTorch
def convert_torch(data, key_list, type_list, device=None):
    '''
    The function of converting data to torch.Tensor.
    
    Parameters
    ----------
    data: dict
        The data.
    key_list: list
        The list of keys.
    type_list: list
        The list of types.
    kwargs: dict, keyword arguments
        Other parameters. (default: {})
    
    Returns
    -------
    data: list
        The converted data to torch.
    '''
    data_list = []
    dtype_dict = {'int': torch.int, 'int32': torch.int32, 'int64': torch.int64, 'long': torch.long,
                  'float': torch.float, 'float32': torch.float32, 'float64': torch.float64}
    for i, key in enumerate(key_list):
        if isinstance(type_list[i], list):
            if device is not None and type_list[i][1] == 'gpu':
                data_s = torch.tensor(data[key], dtype=dtype_dict[type_list[i][0]]).to(device)
            else:
                data_s = torch.tensor(data[key], dtype=dtype_dict[type_list[i][0]])
        else:
            data_s = torch.tensor(data[key], dtype=dtype_dict[type_list[i]])
        data_list.append(data_s)
    return data_list


# Data Loader
def dataloader(dataset, batch_size=32, keylist=['x', 'y'], typelist=None, pin_memory=False, device=None, **kwargs):
    '''
    data loader
        - Load the dataset.
        - Return data loader.

    Parameters
    ----------
    dataset: OneFoldDataSet
        has train and test.
    batch_size: int, optional
        Batch size. (default: 32)
    keylist: list, optional
        The keys list of data. (default: ['x', 'y'])
    typelist: list, optional
        The types list of data. (default: None)
    device: torch.device, optional
        The device. (default: None)
    pin_memory: bool, optional
        Pin memory. (default: False)

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
        Data loader for training data.
    test_loader: torch.utils.data.DataLoader
        Data loader for testing data.
    '''
    # If dataset is dict
    if not isinstance(dataset, OneFoldDataSet):
        raise ValueError("The type of dataset must be OneFoldDataSet.")

    # Detect key list and type list
    data_key_list = dataset.keys()
    for k in keylist:
        if k not in data_key_list:
            raise ValueError(f"The key {k} is not in the dataset.")
    if typelist is None:
        typelist = ['float'] * len(keylist)
    
    # initial data loader
    train_list = convert_torch(dataset.train, keylist, typelist, device)
    test_list = convert_torch(dataset.test, keylist, typelist, device)

    # TensorDataset
    train_dataset = Data.TensorDataset(*train_list)
    test_dataset = Data.TensorDataset(*test_list)

    # DataLoader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader
# %% Main Function
if __name__ == '__main__':
    pass
