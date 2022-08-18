#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   tools.py
@Time    :   2022/07/01 23:25:32
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   tools
'''
#%% Import Packages
# basic
import copy

# %% My Structure (Classes)
class MyStruct():
    """
    A template of structure (class).
    """
    def __init__(self, name=None, content=[]):
        """
        Initial MyStruct

        Parameters
        ----------
        name: str or None, optional
            Structure name. (default: None)
        content: list or tuple, optional
            Content of this structure, which can be shown by print function. (default: [])
        """
        # structure name
        if isinstance(name, str) or (name is None):
            self.__struct_name__ = self.__class__.__name__ if name is None else name
        # content
        if isinstance(content, list) or isinstance(content, tuple):
            pass
        else:
            raise ValueError("The type of argument 'content' should be list or tuple")
        self.__content__ = [] if len(content) == 0 else copy.deepcopy(content)
    
    @property
    def dict(self):
        """
        Return a dict of the struct.
        """
        struct_dict = self.__dict__.copy()
        struct_dict.pop('__struct_name__')
        struct_dict.pop('__content__')
        return struct_dict

    @property
    def content_print(self):
        return "" if len(self.__content__) == 0 else f"{[i for i in self.__content__]}"
   
    def __repr__(self):
        """
        Print
        """
        return self.__struct_name__ + self.content_print # put constructor arguments in the ()


# %% Main Function 
if __name__ == "__main__":
    a = MyStruct()
    b = MyStruct("B")
    c = MyStruct("C", [1, 2, 3])
    c.hello = "hello"
    