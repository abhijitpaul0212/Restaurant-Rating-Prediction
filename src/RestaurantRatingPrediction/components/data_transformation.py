# data_transformation.py

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import CustomException
from src.RestaurantRatingPrediction.utils.utils import Utils
from src.RestaurantRatingPrediction.utils.data_processor import CSVProcessor
from src.RestaurantRatingPrediction.utils.transformer import EncodeTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataTransformationConfig:
    """
    This is configuration class for Data Transformation
    """
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    This class handles Data Transformation
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.utils = Utils()
        self.csv_processor = CSVProcessor()

    def transform_data(self):
        try:
            numerical_features = ['votes']
            categorical_features = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'cost_for_2', 'type']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    
                ])

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories='auto')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ], remainder='passthrough')

            return preprocessor
        
        except Exception as e:
            logging.error("Exception occured in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, val_path=None):
            
        try:
            train_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=train_path)
            test_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=test_path)
            
            logging.info(f'Train Dataframe Head : \n{train_df.head(2).to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head(2).to_string()}')
            
            # Fix `cost` column values
            train_df = self.utils.handle_cost_column(train_df)
            test_df = self.utils.handle_cost_column(test_df)

            logging.info("Handling rate columns")
            train_df = self.utils.handle_rate_column(train_df)
            test_df = self.utils.handle_rate_column(test_df)

            logging.info("Handling categorical columns")
            column_thresholds = {"rest_type": 1000, "cuisines": 300, "location": 500}
            train_df = self.utils.handle_categorical_columns(train_df, column_thresholds)
            test_df = self.utils.handle_categorical_columns(test_df, column_thresholds)

            target_column_name = 'rate'
            drop_columns = [target_column_name, '_id']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training, validation and testing datasets")
            preprocessing_obj = self.transform_data()
            preprocessing_obj.fit(input_feature_train_df)
            
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Converting features and target columns from np.arrays to dataframes")            

            input_feature_train_arr_df = pd.DataFrame(input_feature_train_arr, columns=preprocessing_obj.get_feature_names_out())
            input_feature_test_arr_df = pd.DataFrame(input_feature_test_arr, columns=preprocessing_obj.get_feature_names_out())

            train_df = pd.concat([input_feature_train_arr_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_arr_df, target_feature_test_df], axis=1)

            logging.info(f'Processed Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Processed Test Dataframe Head : \n{test_df.head().to_string()}')
     
            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing pickle file saved")
            
            return (
                train_df,
                test_df
            )

        except Exception as e:
            logging.error("Exception occured in Initiate Data Transformation")
            raise CustomException(e, sys)
