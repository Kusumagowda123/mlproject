# utils file consists of commonly used utility functions such as saving and loading objects by entire project
from copyreg import pickle
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill  # dill is used for serializing and deserializing python objects 
from sklearn.metrics import r2_score  
from sklearn.model_selection import GridSearchCV 
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
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    '''
    This function evaluates multiple machine learning models and returns their R2 scores.
    '''
    try:
        
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            # Train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)