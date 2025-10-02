from dotenv import load_dotenv
load_dotenv()
from dataset_processor import DataSetProcessor
from os import environ
import logging
from sys import exit

PATH_TO_ZIP = environ.get("PATH_TO_ZIP")
DATA_FILE_NAME = environ.get("DATA_FILE_NAME")

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    dp = DataSetProcessor()
    if(not PATH_TO_ZIP or not DATA_FILE_NAME):
        log.error("No path to zip file or data file name provided. Please add a .env file with these values")
        exit(1)
    dp.prepare_data(PATH_TO_ZIP, DATA_FILE_NAME)

