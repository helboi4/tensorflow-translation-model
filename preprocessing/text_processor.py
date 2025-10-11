from typing import Text
from keras.layers import TextVectorization
import tensorflow_text as tf_txt
import tensorflow as tf
import keras
from config.training_config import TrainingConfig
from enums.language import Language
from enums.language_family import LanguageFamily
from mappers.language_mapper import get_language_family
from config.language_config import LanguageConfig, get_language_config

class TextProcessor():

    def __init__(self, lang_config ):
        self.lang_config = lang_config

    def sanitize_text(self, text):
        lang_config = self.lang_config
        #Normalize Unicode
        text = tf_txt.normalize_utf8(text, lang_config.normalization_form)

        #Lowercase languages that have case distinction
        if lang_config.needs_lowercasing:
            text = tf.strings.lower(text)

        #Keep valid characters and punctuation
        keep_pattern = f"[^{lang_config.character_range}{config.punctuation}]"
        text = tf.strings.regex_replace(text, keep_pattern, "")

        #Add spaces around punctuation
        punct_pattern = f"[{lang_config.punctuation}]"
        text = tf.strings.regex_replace(text, punct_pattern, r" \0")

        #Strip whitespace
        text = tf.strings.strip(text)

        #Add START and END toekns
        text = tf.strings.join(["[START]", text, "[END]"], separator=" ")

        return text

    def create_processors(self, train_raw, val_raw):
        lang_config = self.lang_config

        #Text processing for original language (lang_configarable)
        context_text_processor = keras.layers.TextVectorization(
            standardize=self.sanitize_text,
            max_tokens=lang_config.vocab_size,
            ragged=True
        )
        print("here")
        context_text_processor.adapt(train_raw.map(lambda context, target: context))

        #Text processing for the language we're translating into (always English for now)
        target_text_processor = keras.layers.TextVectorization(
            standardize=self.sanitize_text,
            max_tokens=get_language_config(LanguageFamily.LATIN).vocab_size,
            ragged=True
        )
        target_text_processor.adapt(train_raw.map(lambda context, target: target))
        
        return context_text_processor, target_text_processor
        
    def create_datasets(self, context_text_processor, target_text_processor, train_raw, val_raw):
        def create_input_label_pairs(context, target):
            context = context_text_processor(context).to_tensor()
            target = target_text_processor(target)
            targ_in = target[:,:-1].to_tensor()
            targ_out = target[:,1:].to_tensor()
            return (context, targ_in), targ_out

        train_ds = train_raw.map(create_input_label_pairs, tf.data.AUTOTUNE)
        val_ds = val_raw.map(create_input_label_pairs, tf.data.AUTOTUNE)

        return train_ds, val_ds

    def create_training_config(self, train_raw, val_raw):
        context_text_processor, target_text_processor = self.create_processors(train_raw, val_raw)
        train_ds, val_ds = self.create_datasets(context_text_processor, target_text_processor, train_raw, val_raw)
        return TrainingConfig(
            context_text_processor=context_text_processor,
            target_text_processor=target_text_processor,
            train_ds=train_ds,
            val_ds=val_ds
        )

