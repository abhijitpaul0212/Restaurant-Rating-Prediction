# model_trainer.py

import os
import sys
import numpy as np
from dataclasses import dataclass
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import CustomException
from src.RestaurantRatingPrediction.utils.utils import Utils

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    """
    This is configuration class FOR Model Trainer
    """
    trained_model_obj_path: str = os.path.join("artifacts", "model.pkl.gz")
    trained_model_report_path: str = os.path.join('artifacts', 'model_report.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = Utils()

    def initiate_model_training(self, train_dataframe, test_dataframe):
        try:
            logging.info("Splitting Dependent and Independent features from train and validation & test dataset")

            X_train, y_train, X_test, y_test = (
                train_dataframe.iloc[:, :-1],
                train_dataframe.iloc[:, -1],
                test_dataframe.iloc[:, :-1],
                test_dataframe.iloc[:, -1])
            
            models = {
                    'LinearRegression': LinearRegression(),
                    'Lasso': Lasso(),
                    'Ridge': Ridge(),
                    'ElasticNet': ElasticNet(),
                    'RandomForestRegressor': RandomForestRegressor(),
                    'GradientBoostingRegressor': GradientBoostingRegressor(),
                    'ExtraTreesRegressor': ExtraTreesRegressor(),
                    'DecisionTreeRegressor': DecisionTreeRegressor(),
                    'XGBRegressor': XGBRegressor()
                }
            
            # models = {
            #         'XGBRegressor': (XGBRegressor(), {
            #                     'n_estimators': [10, 100, 1000],
            #                     'learning_rate': [0.001, 0.01, 1, 10, 100],
            #                     'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
            #                     'subsample': [0.01, 0.1, 10, 100],
            #                     'colsample_bytree': [0.3, 0.4, 0.5, 0.7]}),
            #         'DecisionTreeRegressor': (DecisionTreeRegressor(), {
            #                              'max_depth': [20, 30, 50, 100], 
            #                              'max_features': [10, 20, 40, 70]
            #                              }),
            #         'ExtraTreesRegressor': (ExtraTreesRegressor(), {
            #                 'n_estimators': [10, 100, 1000],
            #                 'max_depth': [20, 30, 50, 100], 
            #                 'max_features': [10, 20, 40, 70]}),
            #         'GradientBoostingRegressor': (GradientBoostingRegressor(), {
            #                                 'n_estimators': [10, 100, 1000],
            #                                 'learning_rate': [0.001, 0.01, 1, 10, 100],
            #                                 'max_depth': [20, 30, 50, 100]}),
            #         'RandomForestRegressor': (RandomForestRegressor(), {
            #                             'n_estimators': [10, 100, 1000],
            #                             'max_depth': [20, 30, 50, 100], 
            #                             'max_features': [10, 20, 40, 70]})
            #         }

            # model evaluation without any hyper-paramter tuning            
            best_model = self.utils.evaluate_models(models, X_train, y_train, X_test, y_test, metric="r2")
            
            # model evaluation along with hyper-paramter tuning
            # best_model = self.utils.evaluate_models_with_hyperparameter(models, X_train, y_train, X_test, y_test, metric="r2", verbose=3)
            
            self.utils.save_object(
                 file_path=self.model_trainer_config.trained_model_obj_path,
                 obj=best_model
            )       

        except Exception as e:
            raise CustomException(e, sys)
