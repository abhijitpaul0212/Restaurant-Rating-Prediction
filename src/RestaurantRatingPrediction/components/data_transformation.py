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
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, LabelEncoder

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
            numerical_features = ['votes', 'cost_for_two']
            categorical_features = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'menu_item']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    
                ])

            cat_pipeline = Pipeline(
                steps=[
                    ('encoder', EncodeTransformer()),
                    ('imputer', SimpleImputer(strategy='most_frequent')),
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

    def initiate_data_transformation(self, train_path, test_path, val_path):

        def update_column_values(df):
            df = df.replace({'online_order': {'Yes': True, 'No': False}, 
                             'book_table': {'Yes': True, 'No': False}})
            logging.info("Categorical values has been converted to boolean values")
            return df
        
        # Fixing rate column
        def update_rate(df):
            def fix_rate(rate):
                if rate == 'NEW' or rate == '-' or rate == np.nan:
                    return np.nan
                else:
                    return str(rate).split("/")[0]
                
            df['rate'] = pd.to_numeric(df['rate'].apply(fix_rate), errors='coerce')
            logging.info("`rate` column values are fixed")
            return df
        
        # Fixing cost column
        def update_cost(df):
            def fix_cost(cost):
                cost = str(cost).replace(",", "")
                return float(cost)

            df['cost_for_two'] = df['cost_for_two'].apply(fix_cost)
            logging.info("`cost` column values are fixed")
            return df

        def update_column_names(df):
            df = df.rename(columns={'approx_cost(for two people)': 'cost_for_two', 'listed_in(type)': 'type'})
            logging.info("Column names updated")
            return df
        
        def handle_missing_rate(df):
            df['rate'] = df['rate'].fillna(df['rate'].mean())
            logging.info("`rate` column missing values are handled")
            return df
    
        try:
            train_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=train_path)
            test_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=test_path)
            val_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=val_path)
            
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            logging.info(f'Validation Dataframe Head : \n{val_df.head().to_string()}')
            
            # Rename columns
            train_df = update_column_names(train_df)
            test_df = update_column_names(test_df)
            val_df = update_column_names(val_df)

            # Update column values
            train_df = update_column_values(train_df)
            test_df = update_column_values(test_df)
            val_df = update_column_values(val_df)

            # Fix `rate` column values
            train_df = update_rate(train_df)
            test_df = update_rate(test_df)
            val_df = update_rate(val_df)

            # Fix `cost` column values
            train_df = update_cost(train_df)
            test_df = update_cost(test_df)
            val_df = update_cost(val_df)

            # Handle missing `rate` i.e. Target column values
            train_df = handle_missing_rate(train_df)
            test_df = handle_missing_rate(test_df)
            val_df = handle_missing_rate(val_df)

            target_column_name = 'rate'
            drop_columns = [target_column_name, '_id']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_val_df = val_df.drop(columns=drop_columns, axis=1)
            target_feature_val_df = val_df[target_column_name]

            # Apply transformation
            preprocessing_obj = self.transform_data()
            preprocessing_obj.fit(input_feature_train_df)
            
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)
            
            input_feature_train_arr_df = pd.DataFrame(input_feature_train_arr, columns=['votes', 'cost_for_two', 'online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'menu_item'])
            input_feature_test_arr_df = pd.DataFrame(input_feature_test_arr, columns=['votes', 'cost_for_two', 'online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'menu_item'])
            input_feature_val_arr_df = pd.DataFrame(input_feature_val_arr, columns=['votes', 'cost_for_two', 'online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'menu_item'])

            logging.info("Applying preprocessing object on training, vdalidation and testing datasets")

            train_df = pd.concat([input_feature_train_arr_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_arr_df, target_feature_test_df], axis=1)
            val_df = pd.concat([input_feature_val_arr_df, target_feature_val_df], axis=1)

            logging.info(f'Processed Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Processed Test Dataframe Head : \n{test_df.head().to_string()}')
            logging.info(f'Processed Validation Dataframe Head : \n{val_df.head().to_string()}')

            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_df,
                test_df,
                val_df
            )

        except Exception as e:
            logging.error("Exception occured in Initiate Data Transformation")
            raise CustomException(e, sys)
