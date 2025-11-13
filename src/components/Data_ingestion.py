import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging 
import os
import sys
from src.exception import CustomException

@dataclass
class dataingestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion(dataingestionConfig):
    def __init__(self):
        self.ingestion_config = dataingestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read data as DataFrame')
            os.makedirs((os.path.dirname(self.ingestion_config.train_data_path)), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path,header = True,index = False)
            
            logging.info('train test split initiated')
            
            train_set, test_set = train_test_split(df,test_size = 0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, header=True, index= False)
            test_set.to_csv(self.ingestion_config.test_data_path, header=True, index= False)
            logging.info('Data Ingestion completed')
            
            return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
            
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    
