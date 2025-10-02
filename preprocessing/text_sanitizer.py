from typing import Text
import tensorflow_text as tf_txt
import tensorflow as tf
from enums.language import Language
from mappers.language_mapper import get_language_family
from config.text_processing_config import TextProcessingConfig, get_processing_config

class LanguageConfigurationError(Exception):
    pass

class TextSanitizer():

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




