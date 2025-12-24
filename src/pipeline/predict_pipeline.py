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
            preprocessor_path = 'artifacts/preprocessor.pkl'

            logging.info("Loading preprocessor and model")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Transforming features using preprocessor")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions")
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):
        self.gender = gender if gender else 'male'
        self.race = race_ethnicity if race_ethnicity else 'group A'
        self.parental_level_of_education = parental_level_of_education if parental_level_of_education else 'some high school'
        self.lunch = lunch if lunch else 'standard'
        self.test_preparation_course = test_preparation_course if test_preparation_course else 'none'
        self.reading_score = float(reading_score) if reading_score else 50.0
        self.writing_score = float(writing_score) if writing_score else 50.0

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)