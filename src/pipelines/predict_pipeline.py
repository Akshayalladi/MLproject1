import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/processor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            scaled_data = preprocessor.transform(features)
            prediction = model.predict(scaled_data)
            return prediction
        except Exception as e:
            raise CustomException(f"Error while predicting with error: {str(e)}", sys)


class Datapreparation:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def create_dataFrame(self):
        try:
            data_input_dict: dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(data_input_dict)

        except Exception as e:
            raise CustomException(f"Error while preparing DataFrame: {str(e)}", sys)
            
      
            
    
        
        