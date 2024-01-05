# training_pipeline.py

from src.RestaurantRatingPrediction.components.data_ingestion import DataIngestion
from src.RestaurantRatingPrediction.components.data_transformation import DataTransformation
from src.RestaurantRatingPrediction.components.model_trainer import ModelTrainer
from src.RestaurantRatingPrediction.components.model_evaluation import ModelEvaluation
from dataclasses import dataclass
import mlflow
import src.RestaurantRatingPrediction.utils.mlflow_setup as mlflow_setup
from urllib.parse import urlparse


@dataclass
class TrainingPipeline:

    def start(self):

        mlflow.set_registry_uri("https://dagshub.com/abhijitpaul0212/Restaurant-Rating-Prediction.mlflow")
        mlflow_setup.start_mlflow_run()

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)

        model_evaluation = ModelEvaluation()
        model_evaluation.initiate_model_evaluation(test_arr)


if __name__ == '__main__':
    training_pipeline = TrainingPipeline()
    training_pipeline.start()
