from dotenv import load_dotenv
load_dotenv()
from config.training_config import TrainingConfig
from preprocessing.data_loader import DataLoader
from preprocessing.text_processor import TextProcessor
from os import environ
import logging
from sys import exit
import tensorflow as tf
from enums.language import Language
from mappers.language_mapper import get_language_family
from config.language_config import get_language_config
from model.translator import Translator
from exporter.export import Export

PATH_TO_ZIP = environ.get("PATH_TO_ZIP")
DATA_FILE_NAME = environ.get("DATA_FILE_NAME")
LANGUAGE = Language(environ.get("LANGUAGE_CODE"))
LANGUAGE_FAMILY = get_language_family(LANGUAGE)
LANG_CONFIG = get_language_config(LANGUAGE_FAMILY)
UNITS = 256

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

def preprocess() -> TrainingConfig:
    dl = DataLoader()
    if(not PATH_TO_ZIP or not DATA_FILE_NAME):
        log.error("No path to zip file or data file name provided. Please add a .env file with these values")
        exit(1)
    train_raw, val_raw = dl.load_raw_data(PATH_TO_ZIP, DATA_FILE_NAME)

    tp = TextProcessor(LANG_CONFIG)
    return tp.create_training_config(train_raw, val_raw)




if __name__ == "__main__":
    training_config: TrainingConfig = preprocess()
    translator = Translator(UNITS, training_config)
    translator.train()
    export = Export(translator)
    tf.saved_model.save(export, "translator", options=tf.saved_model.SaveOptions(save_debug_info=True))
    
