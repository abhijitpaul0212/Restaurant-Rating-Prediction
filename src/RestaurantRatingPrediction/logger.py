# logger.py

import os
import logging
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

LOG_PATH = os.path.join(os.getcwd(), "logs")

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILEPATH = os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILEPATH,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info("Demo logging activity")
    abc = [1, 2, 3, 4]
    from src.CreditCardDefaultsPrediction.utils.utils import save_object

    save_object(os.path.join('logs', 'loggger.pkl'), abc)
