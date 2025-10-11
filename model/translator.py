import tensorflow as tf
import keras
from config.training_config import TrainingConfig
from model.decoder import Decoder
from model.encoder import Encoder
from config.language_config import get_language_config
from enums.language_family import LanguageFamily
import logging

class Translator(keras.Model):

    def __init__(self, units, config: TrainingConfig):
        super().__init__()
        #Build the encorder and decoder
        encoder = Encoder(config.context_text_processor, units)
        decoder = Decoder(config.target_text_processor, units, get_language_config(LanguageFamily.LATIN))

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
        self.log = logging.getLogger(__name__)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        #TODO: Tutorial has a todo here that says "remove this". I don't know why yet
        try:
            #Delete the keras mask so keras doesn't scale the loss+accuracy
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


    #A loss is a calculation of how far off the model's predicted probabilities for each word were based on the real word token
    #y_pred represents an array for the length of the sentence, with each item being an array of the correct word being one of 5 candidates
    #y_true is an array for the length of the sentence, with each item being the token of the correct word for that index of the sentence
    #If the highest probability in y_pred[i] is not for the token y_true[i] then that increases the loss value (bad!)
    #A mask marks which values in y_true are real words vs padding, 1 for words and 0 for padding
    # so if y_true = [45, 124, 89, 0, 0] and 0 is the token for padding, mask = [1, 1, 1, 0, 0]
    #To calculate the true loss we need to apply a mask to ensure that we only calculate losses from real words (not padding) to maintain accuracy of the loss value
    def masked_loss(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        loss = loss_fn(y_true, y_pred)
        
        #Use mask to remove the loses on padding values
        mask = tf.cast(y_true != 0, loss.dtype)
        loss *= mask

        #Return the mean loss over real words only
        #Mask has value 1 for each real word so the sum of its elements is the number of real words
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def masked_acc(self, y_true, y_pred):
        #Get the highest prediction
        y_pred = tf.argmax(y_pred, axis=-1)
        #Convert to same type for comparison
        y_pred = tf.cast(y_pred, y_true.dtype)
        
        #Creates an array of boolean values representing the accuracy of each word prediction (correct or incorrect)
        match = tf.cast(y_true == y_pred, tf.float32)
        mask = tf.cast(y_true != 0, tf.float32)

        #Return mean accuracy over real words only
        return tf.reduce_sum(match) /tf.reduce_sum(mask)

    def train(self):
        model = self
        context_text_processor = self.config.context_text_processor
        target_text_processor = self.config.target_text_processor
        train_ds = self.config.train_ds
        val_ds = self.config.val_ds

        model.compile(
            optimizer="adam",
            loss=model.masked_loss,
            metrics=[model.masked_acc, model.masked_loss]
        )

        model.fit(
            train_ds.repeat(),
            epochs=100,
            steps_per_epoch=100,
            validation_data=val_ds,
            validation_steps=20,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3)
            ]
        )

    def translate(self,
                  texts, *,
                  max_length=50,
                  temperature=0.0):
      # Process the input texts
      context = self.encoder.convert_input(texts)
      batch_size = tf.shape(texts)[0]
    
      # Setup the loop inputs
      tokens = []
      attention_weights = []
      next_token, done, state = self.decoder.get_initial_state(context)
    
      for _ in range(max_length):
        # Generate the next token
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done,  state, temperature)
    
        # Collect the generated tokens
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)
    
        if tf.executing_eagerly() and tf.reduce_all(done):
          break
    
      # Stack the lists of tokens and attention weights.
      tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
      self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)
    
      result = self.decoder.tokens_to_text(tokens)
      return result
