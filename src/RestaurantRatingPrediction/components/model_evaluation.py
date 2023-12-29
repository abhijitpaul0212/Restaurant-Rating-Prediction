# model_evaluation.py

import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from urllib.parse import urlparse
from dataclasses import dataclass

from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import CustomException
from src.RestaurantRatingPrediction.utils.utils import Utils
from src.RestaurantRatingPrediction.utils.mlflow_setup import setup_mlflow_experiment
import src.RestaurantRatingPrediction.utils.mlflow_setup as mlflow_setup


@dataclass
class ModelEvaluation:

    def eval_metrics(self, model, features, label):

        r2 = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='r2').mean(), 2)
        mse = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='neg_mean_squared_error').mean(), 2)
        rmse = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring=make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)).mean(), 2)
        return r2, mse, rmse
    
    def initiate_model_evaluation(self, test_array):
        try:
            X_test, y_test = (test_array.iloc[:, :-1], test_array.iloc[:, -1])
            model_path = os.path.join("artifacts", "model.pkl.gz")
            model = Utils().load_object(model_path)

            with mlflow.start_run(run_id=mlflow_setup.get_active_run_id(), nested=True):
                
                mlflow.set_tag("Best Model", str(model).split("(")[0])
                
                (r2, mse, rmse) = self.eval_metrics(model, X_test, y_test)

                logging.info("r2_score: {}".format(r2))
                logging.info("mse_score: {}".format(mse))
                logging.info("rmse_score: {}".format(rmse))

                mlflow.log_metric("r2 score", r2)
                mlflow.log_metric("mse score", mse)
                mlflow.log_metric("rmse score", rmse)
                mlflow.end_run()

                mlflow.sklearn.log_model(model, "model")
                
        except Exception as e:
            raise CustomException(e, sys)
