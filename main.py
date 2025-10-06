from dotenv import load_dotenv
load_dotenv()
from preprocessing.data_loader import DataLoader
from preprocessing.text_processor import TextProcessor
from os import environ
import logging
from sys import exit

PATH_TO_ZIP = environ.get("PATH_TO_ZIP")
DATA_FILE_NAME = environ.get("DATA_FILE_NAME")
LANGUAGE_CODE = environ.get("LANGUAGE_CODE")

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    dl = DataLoader()
    if(not PATH_TO_ZIP or not DATA_FILE_NAME):
        log.error("No path to zip file or data file name provided. Please add a .env file with these values")
        exit(1)
    train_raw, val_raw = dl.load_raw_data(PATH_TO_ZIP, DATA_FILE_NAME)
    tp = TextProcessor(LANGUAGE_CODE)
