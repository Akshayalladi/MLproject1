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
            
            param_grid = {
                            "Random Forest": {
                                'n_estimators': [100, 200],
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5],
                                'max_features': ['auto', 'sqrt']
                            },
                            "Decision Tree": {
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5],
                                'criterion': ['squared_error', 'friedman_mse']
                            },
                            "Gradient Boosting": {
                                'n_estimators': [100, 200],
                                'learning_rate': [0.01, 0.1],
                                'max_depth': [3, 5],
                                'subsample': [0.8, 1.0]
                            },
                            "Linear Regression": {
                                'fit_intercept': [True, False]
                                
                            },
                            "XGBRegressor": {
                                'n_estimators': [100, 200],
                                'learning_rate': [0.01, 0.1],
                                'max_depth': [3, 5],
                                'subsample': [0.8, 1.0],
                                'colsample_bytree': [0.8, 1.0]
                            },
                            "CatBoosting Regressor": {
                                'iterations': [100, 200],
                                'learning_rate': [0.01, 0.1],
                                'depth': [4, 6],
                                'l2_leaf_reg': [1, 3]
                            },
                            "AdaBoost Regressor": {
                                'n_estimators': [50, 100],
                                'learning_rate': [0.01, 0.1, 1.0],
                                'loss': ['linear', 'square', 'exponential']
                            }
                    }
            model_report: dict = evaluate_model(x_train=x_train,y_train = y_train, x_test= x_test,y_test= y_test, models= models, param_grid= param_grid)
           
            
            best_model_score = max([v.get('best_score', float('-inf')) for v in model_report.values()])
          
            best_model_name = max(model_report.items(), key=lambda item: item[1].get('best_score', float('-inf')))[0]

            # Use the trained best estimator from the report if available, otherwise fall back to the model in `models`
            models_best = model_report[best_model_name].get('best_estimator', models.get(best_model_name))
            if best_model_score < 0.6:
                raise CustomException(f"No best model found", sys)
            
            logging.info(f"Best model found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            save_object(
                file_path=self.modelTrainerConfig.model_file_path, 
                obj=models_best
            )
            
            
            predicted = models_best.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square,best_model_name
            
        except Exception as e:
            raise CustomException(f"Error occured while initiating model trainer {str(e)}",sys)