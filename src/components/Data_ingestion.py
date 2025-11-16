import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.Data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import modelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion(DataIngestionConfig):
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Use forward slashes (works on Windows with pandas) or os.path.join
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read data as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)

            logging.info("train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, header=True, index=False)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(f"Error while initiating data ingestion with error message: {str(e)}", sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    r2_square, best_model_name = model_trainer.initiate_model_trainer(train_array, test_array)
    print(f"R2 square value: {r2_square}")

