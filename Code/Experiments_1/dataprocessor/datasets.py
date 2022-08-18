#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2022/06/21 22:28:40
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   datasets
'''

# %% Import Packages
# Basic
import torch
import torch.utils.data as Data

# Modules
if __package__ is None:
    import process
else:
    from . import process

# %% Classes


# %% Functions
def load_dataset(dataset_name, train_ratio=0.8, cv=1, seed=None, **kwargs):
    '''
    The function of splitting data which can: 
        - Load the dataset.
        - Preprocess the data.
        - Return train/test split.
    Parameters
    ----------
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
        The dataset with cross-validation.
    '''
    # Load data
    data = load(dataset_name=dataset_name, seed=seed, **kwargs)
    # Split data
    dataset = process.data_split(data, dataset_name, train_ratio=train_ratio, cv=cv, seed=seed, **kwargs)
    return dataset

def load(dataset_name, seed=None,  **kwargs):
    """
    The function of loading datasets which can: 
        - Load the dataset.
        - Preprocess the data.
        - Return train/test split.

    Parameters
    ----------
    dataset_name: str
        Dataset name.
    seed: int, optional
        Random seed. (default: None)
    kwargs: dict, keyword arguments
        Other parameters. (default: {})

    Returns
    -------
    data: dict
    """
    # If dataset_name is string
    if not isinstance(dataset_name, str):
        raise ValueError("The type of dataset_name must be a string.")

    # Find dataset
    dataset_name_lower = dataset_name.lower().strip()
    if dataset_name_lower == "synthetic":
        from .dataset_Synthetic import load as load_synthetic
        data_orginal = load_synthetic(random_seed=seed, **kwargs)
    elif dataset_name_lower == "":
        pass
    else:
        raise ValueError(f"There is no dataset called '{dataset_name}'.")

    return data_orginal


# %% Main Function
if __name__ == "__main__":
    print('datasets')
