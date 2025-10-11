import keras
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    context_text_processor: keras.layers.TextVectorization
    target_text_processor: keras.layers.TextVectorization
    train_ds: np.ndarray
    val_ds: np.ndarray
