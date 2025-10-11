import keras
import numpy as np

class TrainingConfig:
    context_text_processor: keras.layers.TextVectorization
    target_text_processor: keras.layers.TextVectorization
    train_ds: np.ndarray
    val_s: np.ndarray
