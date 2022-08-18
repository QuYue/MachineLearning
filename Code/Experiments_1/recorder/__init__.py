#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2022/07/15 05:07:34
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   __init__ of recorder
'''

# %% Import Packages
# Basic
import os
import sys
import json
import datetime
import copy

# Add path
if __package__ is None:
    os.chdir(os.path.dirname(__file__))
    sys.path.append("..")

# Self-defined
from . import record
from . import parameter
from . import analysis

# %%
