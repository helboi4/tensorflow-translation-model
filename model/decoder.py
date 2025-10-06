import tensorflow as tf
import keras
from cross_attention import CrossAttention
from utils.shape_checker import ShapeChecker

#The decoder is used to go through the English sentence one word at a time and predict the next word
#For this reason, it is unidirectional instead of bidirectional like the encoder
#since it is used more for prediction tha underastanding the context like the encoder is
class Decoder(keras.layers.Layer):
    
    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        #The oov token is the token used for when the word is not in the supplied vocabulary
        self.word_to_id = keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token="", oov_token="[UNK]",
        )
        self.id_to_word = keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token="", oov_token="[UNK]",
            invert=True
        )
        self.start_token = self.word_to_id("[START]")
        self.end_token = self.word_to_id("[END]")
        
        self.units = units

        #Converts token ids to vectors
        self.embedding = keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        #RNN keeps track of what's been generated so far
        self.rnn = keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer="glorot_uniform")

        #The RNN output is used as the query for the attention layer
        self.attention = CrossAttention(units)

        #Logits are a vector of non-normalized predictions
        #This layer produces the logits for each output token
        self.output_layer = keras.layers.Dense(self.vocab_size)
   
    #Called by the super __call__ dunder method
    #context = the foreign langaue vectors from the encoder's output
    #x = the English sequence input
    #state (optional) = the previous state output from the decoder's RNN
    #You can pass the state from a previous run to continue generating text where you left off
    #return_state (optional) = set this to true to return the state to pick up later as above
    def call(self, context, x, state=None, return_state=False):
        shape_checker = ShapeChecker()
        shape_checker(x, "batch t")
        shape_checker(context, "batch s units")

        #Lookup the embeddings (vectors)
        x = self.embedding(x)
        shape_checker(x, "batch t units")

        #Process the target sequence
        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, "batch t units")

        #Use the RNN output as the query for the attention
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, "batch t units")
        shape_checker(self.last_attention_weights, "batch t s")

        #Generate logit predictions for the next token
        logits = self.output_layer(x)
        shape_checker(logits, "batch t target_vocab_size")

        if return_state:
            return logits, state
        else:
            return logits

    #Methods below are for use in inference after training

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], self.start_token)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        #TODO: work out if this needs to also be adjusted based on the language config
        result = tf.strings.reduce_join(words, axis=1, separator=" ")
