import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor
)
from src.utils import evaluate_model
from src.utils import save_object
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class modelTrainerConfig:
    model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer(modelTrainerConfig):
    def __init__(self):
        self.modelTrainerConfig = modelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and testing input data')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report: dict = evaluate_model(x_train=x_train,y_train = y_train, x_test= x_test,y_test= y_test, models= models)
            print(model_report)
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            models_best = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException(Exception("No best model found"), sys)
            
            logging.info(f"Best model found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            save_object(
                file_path=self.modelTrainerConfig.model_file_path, 
                obj=models_best
            )
            
            
            predicted = models_best.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square,best_model_name
            
        except Exception as e:
            raise CustomException(e,sys)