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

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path,mode = 'wb') as file_object:
            pickle.dump(obj, file_object)
        logging.info(f'{file_path} saved successfully')

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train, x_test,y_test, models)-> Dict[str,Any]:
    try:
        model_report = {}
        
        for item in models.items():
            model = item[1]
            model.fit(x_train,y_train)
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)
            r2Scorevalue_train = r2_score(y_test,y_test_pred)
            r2Scorevalue_test = r2_score(y_train,y_train_pred)
            model_report[item[0]]= r2Scorevalue_test

        return model_report
    except Exception as e:
        raise CustomException(e, sys)