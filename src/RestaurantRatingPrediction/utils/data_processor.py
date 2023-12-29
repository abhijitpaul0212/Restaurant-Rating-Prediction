# data_processor.py
import pandas as pd
import os
from pathlib import Path
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.utils.utils import Utils


class DataProcessor:
    def process_data(self, path: str, filename: str, **kwargs):
        pass


class CSVProcessor(DataProcessor):
    def process_data(self, path: str, filename: str, **kwargs):
        logging.info("CSV dataset loaded sucessfully")
        file_path = Path(os.path.join(path, filename)) if path is not None else filename
        return pd.read_csv(file_path, skiprows=kwargs.get('skiprows', 0), skipinitialspace=kwargs.get("skipinitialspace", False))


class JSONProcessor(DataProcessor):
    def process_data(self, path: str, filename: str, **kwargs):
        logging.info("JSON dataset loaded sucessfully")
        return pd.read_json(Path(os.path.join(path, filename)))


class DBProcessor(DataProcessor):
    def process_data(self, uri: str, collection: str, **kwargs):
        data = Utils().get_data_from_database(uri, collection)
        logging.info("Dataset from Database loaded sucessfully")
        return data
