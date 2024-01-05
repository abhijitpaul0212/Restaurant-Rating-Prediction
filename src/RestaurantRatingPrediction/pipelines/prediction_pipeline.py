import os
import sys
import pandas as pd
from src.RestaurantRatingPrediction.exception import CustomException
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.utils.utils import Utils


class PredictPipeline:
    def __init__(self):
        self.utils = Utils()
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl.gz")
            
            preprocessor = self.utils.load_object(preprocessor_path)
            model = self.utils.load_object(model_path)
            
            scaled_data = preprocessor.transform(features)            
            pred = model.predict(scaled_data)
            
            return pred
            
        except Exception as e:
            raise CustomException(e, sys)
    
    
class CustomData:
    def __init__(self,
                 online_order: str,
                 book_table: str,
                 votes: float,
                 location: str,
                 rest_type: str,
                 cuisines: str,
                 cost_for_2: float,
                 type: str
                 ):
        
        self.online_order = online_order
        self.book_table = book_table
        self.votes = votes
        self.location = location
        self.rest_type = rest_type
        self.cuisines = cuisines
        self.cost_for_2 = cost_for_2
        self.type = type            
                
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'online_order': [self.online_order], 
                'book_table': [self.book_table], 
                'votes': [self.votes], 
                'location': [self.location], 
                'rest_type': [self.rest_type], 
                'cuisines': [self.cuisines],
                'cost_for_2': [self.cost_for_2],
                'type': [self.type]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)