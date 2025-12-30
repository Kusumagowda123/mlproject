# utils file consists of commonly used utility functions such as saving and loading objects by entire project
from copyreg import pickle
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill  # dill is used for serializing and deserializing python objects    
def save_object(file_path, obj):
    '''
    Save a python object to a file
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)