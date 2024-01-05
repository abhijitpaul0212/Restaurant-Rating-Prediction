# utils.py
from cProfile import label
import gzip
import mlflow
import mlflow.sklearn
import pickle
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
from pymongo.mongo_client import MongoClient
from src.RestaurantRatingPrediction.exception import CustomException
from src.RestaurantRatingPrediction.logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, make_scorer, mean_squared_error
from imblearn.over_sampling import SMOTE
from src.RestaurantRatingPrediction.utils.mlflow_setup import setup_mlflow_experiment
import src.RestaurantRatingPrediction.utils.mlflow_setup as mlflow_setup


class Utils:

    def __init__(self) -> None:
        self.MODEL_REPORT = {}

    def save_object(self, file_path: str, obj):
        """
        The save_object function saves an object to a file.

        :param file_path: str: Specify the path where the object will be saved
        :param obj: obj: Pass the object to be saved
        :return: None
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with gzip.open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
                logging.info(f"File is saved at '{file_path}' successfully.")
        except Exception as e:
            logging.error("Exception occured during saving object")
            raise CustomException(e, sys)

    def load_object(self, file_path: str):
        """
        The load_object function loads a pickled object from the file_path.

        :param file_path: str: Specify the path of the file that we want to load
        :return: None
        """
        try:
            with gzip.open(file_path, "rb") as file_obj:
                logging.info(f"File at '{file_path}' has been successfully loaded.")
                return pickle.load(file_obj)                
        except Exception as e:
            logging.error("Exception occured during loading object")
            raise CustomException(e, sys)

    def delete_object(self, file_path: str):
        """
        The delete_object function deletes a file from the local filesystem.
        
        :param file_path: str: Specify the path of the file to be deleted
        :return: None
        """
        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Remove the file
                os.remove(file_path)
                logging.info(f"File at '{file_path}' has been successfully deleted.")
            else:
                logging.info(f"File at '{file_path}' does not exist.")
        except Exception as e:
            logging.error("Exception occured during deleting object")
            raise CustomException(e, sys)

    def run_data_pipeline(self, data_processor, filepath: str = None, filename: str = None, **kwargs):
        """
        The run_data_pipeline function is a wrapper function that takes in the data_processor object and 
            calls its process_data method.
        
        :param data_processor: obj: Pass in the data processor class that will be used to process the data
        :param filepath: str: Specify the path to the data file
        :param filename: str: Specify the name of the file that will be used for processing
        :param **kwargs: Pass a variable number of keyword arguments to the function
        :return: The processed data
        """
        return data_processor.process_data(filepath, filename, **kwargs)
    
    def timer(self, start_time=None):
        """
        The timer function is a simple function that takes in an optional start_time argument. 
        If no start_time is provided, the current time will be returned. If a start_time is provided, 
        the difference between the current time and the given start_time will be printed.
        
        :param start_time: Datetime: Start the timer
        :return: None
        """
        
        if not start_time:
            start_time = datetime.now()
            return start_time
        
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            logging.info("Model took: {} hours {} minutes and {} seconds".format(thour, tmin, round(tsec, 2)))

    def predict(self, model_name, model, features, label):
        """
        The predict function predicts the labels using the model and features provided.

        :param model: Pass the model to be used for prediction
        :param features: DataFrame: Features to be used for prediction
        :param label: DataFrame: Label to be used for prediction
        :return: dict: A dictionary with the model, accuracy score, f-score, precision score and recall score
        """
        r2 = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='r2').mean(), 2)
        mse = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='neg_mean_squared_error').mean(), 2)
        rmse = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring=make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)).mean(), 2)

        self.MODEL_REPORT[model_name] = {
            'model': model,
            'r2': r2,
            'mse': mse,
            'rmse': rmse}
        
    def evaluate_models_with_hyperparameter(self, models: dict, train_features, train_label, test_features, test_label, metric='r2', verbose=0):
        """
        The evaluate_models function takes in a tuple of models and their parameters, 
        train_features, train_label, val_features and val_label. It then uses the RandomizedSearchCV function to find the best model for each model passed into it.
        
        :param models: tuple: Models and their parameters
        :param train_features: DataFrame: Training features to the evaluate_models function
        :param train_label: DataFrame: Trtaining labels to the predict function
        :param val_features: DataFrame: Validation features to the evaluate_models function
        :param val_label: Validation labels to the predict function
        :return: tuple: The best model and a dictionary of the model report
        """               
        TRAINING_SCORE = {}

        def log_params_with_prefix(prefix, params):
            """Log parameters with a given prefix to avoid conflicts."""
            for key, value in params.items():
                log_key = f"{prefix}_{key}"
                mlflow.log_param(log_key, value)
                
        for model_name, (model, params) in models.items():
            np.random.seed(42)  
            with mlflow.start_run(run_id=mlflow_setup.get_active_run_id(), nested=True):               
                logging.info("\n\n========================= {} =======================".format(model_name))

                search_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5, scoring=metric, n_jobs=-1, cv=5, verbose=verbose, random_state=5)
                # search_cv = GridSearchCV(estimator=model, param_grid=params, scoring=metric, n_jobs=-1, cv=5, verbose=verbose)

                start_time = self.timer(None)
                search_cv.fit(train_features, train_label)
                self.timer(start_time)

                logging.info("BEST PARAMS: {}".format(search_cv.best_params_))
                log_params_with_prefix(model_name, search_cv.best_params_)

                logging.info("BEST TRAINING SCORE USING HYPER-PARAMTERS: {}".format(round(search_cv.best_score_, 2)))
                TRAINING_SCORE[model_name] = round(search_cv.best_score_, 2)

                mlflow.log_metric("{} score".format(metric), round(search_cv.best_score_, 2))

                self.predict(model_name=model_name, model=search_cv.best_estimator_, features=test_features, label=test_label)
                
        logging.info("All training scores: {}".format(TRAINING_SCORE))
        logging.info("All testing scores: {}".format(self.MODEL_REPORT))

        # Fetching the best of training scores
        SCORES = []
        for model_name, values in self.MODEL_REPORT.items():
            for metric_name, score in values.items():
                if metric_name == metric:
                    SCORES.append((model_name, score))        

        best_score = sorted(SCORES, reverse=True, key=lambda x: x[1])[0][1]
        best_model_name = sorted(SCORES, reverse=True, key=lambda x: x[1])[0][0]                
        best_model = [values['model'] for model_name, values in self.MODEL_REPORT.items() if model_name == best_model_name][0]

        logging.info("BEST MODEL: {}".format(best_model_name))
        logging.info("BEST SCORE: {}".format(best_score))

        return best_model
    
    def evaluate_models(self, models: dict, train_features, train_label, test_features, test_label, metric='r2'):
        """
        The evaluate_models function takes in a tuple of models and their parameters, 
        train_features, train_label, val_features and val_label. It then uses the RandomizedSearchCV function to find the best model for each model passed into it.
        
        :param models: tuple: Models and their parameters
        :param train_features: DataFrame: Training features to the evaluate_models function
        :param train_label: DataFrame: Trtaining labels to the predict function
        :param val_features: DataFrame: Validation features to the evaluate_models function
        :param val_label: Validation labels to the predict function
        :return: tuple: The best model and a dictionary of the model report
        """ 
        np.random.seed(42)        
        self.MODEL_REPORT = {}
        for model_name, model in models.items():            
            logging.info("\n\n========================= {} =======================".format(model_name))

            start_time = self.timer(None)
            model.fit(train_features, train_label)
            self.timer(start_time)

            # Evaluate the best model on the train & test set
            self.predict(model_name=model_name, model=model, features=test_features, label=test_label)
            
        logging.info("All Model Report: {}".format(self.MODEL_REPORT))
        best_model_score = max(sorted(model[metric] for model in self.MODEL_REPORT.values()))
        best_model_name = list(self.MODEL_REPORT.keys())[list(model[metric] for model in self.MODEL_REPORT.values()).index(best_model_score)]
        best_model = self.MODEL_REPORT[best_model_name]['model']
        model_report = self.MODEL_REPORT[best_model_name]
        logging.info("Best Model Report: {} ".format(model_report))
        return best_model

    def smote_balance(self, data):
        """
        The smote_balance function takes in a dataframe and returns the same dataframe with SMOTE resampling applied.
        
        :param data: DataFrame: Pass in the dataframe
        :return: DataFrame: Dataframe with the same number of rows as the original dataset, but now there are an equal number of 0s and 1s in the target column
        """
        
        target_column_name = 'DEFAULT_PAYMENT'
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        
        logging.info('Dataset shape prior resampling: {}'.format(data.shape[0]))
        X_resampled, y_resampled = sm.fit_resample(X=data.drop(columns=target_column_name), y=data[target_column_name])
        data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        logging.info('Dataset shape after resampling: {}'.format(data.shape[0]))
        return data

    def connect_database(self, uri):
        """
        The connect_database function establishes a connection to the MongoDB database.
        
        :param uri: str: URI of mongodb atlas database
        :return: MongoClient: A mongoclient object
        """       
        # uri = "mongodb+srv://root:root@cluster0.k3s4vuf.mongodb.net/?retryWrites=true&w=majority&ssl=true"

        client = MongoClient(uri)
        try:
            client.admin.command('ping')
            logging.info("MongoDb connection established successfully")
            return client
        except Exception as e:
            logging.error("Exception occured during creating database connection")
            raise CustomException(e, sys)

    def get_data_from_database(self, uri, collection):
        """
        The get_data_from_database function takes in a uri and collection name, connects to the database, 
        and returns a pandas dataframe of the data from that collection.

        :param uri: str: MongoDB database URI
        :param collection: str: Database name along with Collection e.g. "credit_card_defaults/data"
        :return: DataFrame: A pandas dataframe
        """
        collection = collection.split("/")
        client = self.connect_database(uri)
        collection = client[collection[0]][collection[1]]
        data = list(collection.find())
        return pd.DataFrame(data)

    def handle_categorical_columns(self, data, column_thresholds):
        try:
            for column_name, threshold in column_thresholds.items():
                column_count = data[column_name].value_counts()
                categories_below_threshold = column_count[column_count < threshold].index
                data[column_name] = np.where(data[column_name].isin(categories_below_threshold), 'others', data[column_name])
                logging.info("Updated column {} with threshold {}".format(column_name, threshold))
            return data

        except Exception as e:
            logging.error("Error in handling categorical columns")
            raise CustomException(e, sys)

    def handle_rate_column(self, data, column_name="rate"):
        try:
            data[column_name] = data[column_name].apply(lambda value: np.nan if value in ["NEW", "-"] else float(str(value).split("/")[0]))

            # Replacing null values with the mean
            data[column_name] = data[column_name].fillna(data[column_name].mean())
            return data

        except Exception as e:
            logging.error("Error in handling {} column".format(column_name))
            raise CustomException(e, sys)
    
    def handle_cost_column(self, data):
        def fix_cost(cost):
            cost = str(cost).replace(",", "")
            return float(cost)

        data['cost_for_2'] = data['cost_for_2'].apply(fix_cost)
        logging.info("`cost` column values are fixed")
        return data


if __name__ == "__main__":
    logging.info("Demo logging activity")

    utils = Utils()
    utils.save_object(os.path.join('logs', 'utils.pkl'), utils)
    utils.load_object(os.path.join('logs', 'utils.pkl'))
    utils.delete_object(os.path.join('logs', 'utils.pkl'))
