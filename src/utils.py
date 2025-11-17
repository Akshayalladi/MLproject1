import sys
import os
import pandas as pd
import numpy as np
import dill
import pickle
from src.exception import CustomException
from src.logger import logging
from typing import Dict, Any
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path,mode = 'wb') as file_object:
            pickle.dump(obj, file_object)
        logging.info(f'{file_path} saved successfully')

    except Exception as e:
        raise CustomException(f"Error while saving {file_path} with error message: {str(e)}",sys)
    
def evaluate_model(x_train,y_train, x_test,y_test, models, param_grid)-> Dict[str,Any]:
    try:
        model_report = {}
        
        for name ,model in models.items():
            try:
                print(f"Running GridSearchCV for {name}...")
                clf = GridSearchCV(model, param_grid[name], cv=5, scoring='r2', n_jobs=-1)
                clf.fit(x_train, y_train)

                model_report[name] = {
                    'ModelObject': clf.best_estimator_,
                    'best_score': clf.best_score_,
                    'best_params': clf.best_params_,
                    'best_estimator': clf.best_estimator_
                }
            except Exception as e:
                raise CustomException(f"Error occurred while tuning {name} with error message: {str(e)}",sys)
                

        return model_report

    except Exception as e:
        raise CustomException(f"Error occurred while evaluateing Model with error message: {str(e)}", sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
            
    except Exception as e:
        raise CustomException(f"Error while loading {file_path} with error: {str(e)}",sys)
    