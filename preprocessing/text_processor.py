from typing import Text
from keras.layers import TextVectorization
import tensorflow_text as tf_txt
import tensorflow as tf
import keras
from enums.language import Language
from enums.language_family import LanguageFamily
from mappers.language_mapper import get_language_family
from config.text_processing_config import TextProcessingConfig, get_processing_config

class LanguageConfigurationError(Exception):
    pass

class TextProcessor():

    def __init__(self, language_code: str ):
        self.language_code = language_code
        self.config = self.get_config()

    def get_config(self) -> TextProcessingConfig:
        if self.language_code not in Language:
            raise LanguageConfigurationError("Invalid language code provided, please fix in .env")
        language_family = get_language_family(Language(self.language_code))
        return get_processing_config(language_family)


    def sanitize_text(self, text):
        config = self.config
        #Normalize Unicode
        text = tf_txt.normalize_utf8(text, config.normalization_form)

        #Lowercase languages that have case distinction
        if config.needs_lowercasing:
            text = tf.strings.lower(text)

        #Keep valid characters and punctuation
        keep_pattern = f"[^{config.character_range}{config.punctuation}]"
        text = tf.strings.regex_replace(text, keep_pattern, "")

        #Add spaces around punctuation
        punct_pattern = f"[{config.punctuation}]"
        text = tf.strings.regex_replace(text, punct_pattern, r" \0")

        #Strip whitespace
        text = tf.strings.strip(text)

        #Add START and END toekns
        text = tf.strings.join(["[START]", text, "[END]"], separator=" ")

        return text

    def process_text(self, train_raw, val_raw):
        config = self.config

        #Text processing for original language (configarable)
        context_text_processor = keras.layers.TextVectorization(
            standardize=self.sanitize_text,
            max_tokens=config.vocab_size,
            ragged=True
        )
        print("here")
        context_text_processor.adapt(train_raw.map(lambda context, target: context))

        #Text processing for the language we're translating into (always English for now)
        target_text_processor = keras.layers.TextVectorization(
            standardize=self.sanitize_text,
            max_tokens=get_processing_config(LanguageFamily.LATIN).vocab_size,
            ragged=True
        )
        target_text_processor.adapt(train_raw.map(lambda context, target: target))

        #
        def create_input_label_pairs(context, target):
            context = context_text_processor(context).to_tensor()
            target = target_text_processor(target)
            targ_in = target[:,:-1].to_tensor()
            targ_out = target[:,1:].to_tensor()
            return (context, targ_in), targ_out

        train_ds = train_raw.map(create_input_label_pairs, tf.data.AUTOTUNE)
        val_ds = val_raw.map(create_input_label_pairs, tf.data.AUTOTUNE)

        return train_ds, val_ds

