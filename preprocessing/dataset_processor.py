from pathlib import Path
from zipfile import ZipFile
import numpy as np
import tensorflow as tf
import logging
from preprocessing.text_processor import TextProcessor
from os import environ

class DataSetProcessor():

    def __init__(self):
        self.file_path = None
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
        self.log = logging.getLogger(__name__)

    def prepare_data(self, path_to_zip:str, data_file_name:str) -> None:
        self.unzip_file_and_set_path(path_to_zip, data_file_name)
        target, context = self.load_data()
        train_raw, val_raw = self.create_tf_dataset(target, context)
        print("train length: " + str(len(train_raw)))
        print("val length: " + str(len(val_raw)))
        self.print_samples_for_debug(train_raw)
        example_text = tf.constant("武器の取り引きなども制限されます。")
        language_code = environ.get("LANGUAGE_CODE")
        text_processor = TextProcessor(language_code=language_code)
        print(text_processor.sanitize_text(example_text).numpy().decode())
        train_ds, val_ds = text_processor.process_text(train_raw, val_raw)
#        for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
#            print(ex_context_tok[0, :10].numpy()) 
#            print()
#            print(ex_tar_in[0, :10].numpy()) 
#            print(ex_tar_out[0, :10].numpy())
        print("Done")

    def unzip_file_and_set_path(self, path_to_zip:str, data_file_name:str) -> None:
        extract_dir = Path("./dataset")
        final_file_path = extract_dir / "data_set.txt"
        if final_file_path.exists():
            self.file_path = final_file_path
            self.log.debug(f"File already exists at {final_file_path}")
            return

        data_file_name_base = data_file_name.split(".")[0]
        data_file_name = f"{data_file_name_base}.txt"

        with ZipFile(path_to_zip, "r") as z:
            z.extract(data_file_name, path=extract_dir)

        extracted_file = extract_dir / data_file_name

        if extracted_file != final_file_path:
            extracted_file.rename(final_file_path)

        self.file_path = final_file_path
        self.log.info(f"Sucessfully unzipped file and set path to {final_file_path}")

    def load_data(self):
        if(self.file_path is None or not Path(self.file_path).exists()):
            self.log.error(f"No file under path '{self.file_path}'")
            raise FileNotFoundError("No data set file loaded")

        text = self.file_path.read_text(encoding='utf-8')
        lines = text.splitlines()
        #the text and translation are split by tabs \t
        sets = [line.split('\t') for line in lines]
        context = np.array([context for target, context, info in sets])
        target = np.array([target for target, context, info in sets])

        return target, context

    def create_tf_dataset(self, target_raw, context_raw):
        BUFFER_SIZE = len(context_raw)
        BATCH_SIZE=64

        is_train = np.random.uniform(size=(len(target_raw), )) < 0.8

        train_raw = (
            tf.data.Dataset
            .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
        )
        val_raw = (
            tf.data.Dataset
            .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
        )
        return train_raw, val_raw

    def print_samples_for_debug(self, train_raw):
        for example_context_strings, example_target_strings in train_raw.take(1):
            japanese_texts = [s.decode('utf-8') for s in example_context_strings.numpy()]
            english_texts = [s.decode('utf-8') for s in example_target_strings.numpy()]
            print("Japanese:")
            for text in japanese_texts[:5]:
                print(f"    {text}")
            print("\nEnglish:")
            for text in english_texts[:5]:
                print(f"    {text}")
            break
