import keras
import tensorflow as tf
from utils.shape_checker import ShapeChecker

class Encoder(keras.layers.Layer):

    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        #Converts tokens to vectors
        #Theses vectors represent where in the neural network the words are in relation to other words
        #Think of it as a big spider diagram where similar words will be close to each other
        #by virtue of thier vector values being similar
        self.embedding = keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        #RNN (recurrent neural network) layer processes those vectors sequentially
        #Because it is bidirectional, it reads from both the left and right of the sentence
        #This means it understands context on either side of the word, like the difference between "river bank" and "bank account"
        self.rnn = keras.layers.Bidirectional(
            merge_mode="sum",
            #GRU = Gated Recurrent Unit  - adds the feature of deciding how much of the context to take into account
            layer=keras.layers.GRU(
                units,
                return_sequences=True,
                recurrent_initializer="glorot_uniform"
            )
        )

    #This is called by keras.layers.Layer __call__ dunder method
    #Input x is tokens
    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, "batch s")

        #Get the vector for each token from the embedding layer
        x = self.embedding(x)
        shape_checker(x, "batch s units")

        #GRU takes the embeddings for each word and changes them to to reflect the context
        x = self.rnn(x)
        shape_checker(x, "batch s units")

        #Returns new sequence of embeddings
        return x

    #Function to handle raw strings that have not yet been tokenised
    def convert_input(self, texts):

        texts = tf.convert_to_tensor(texts)

        #If we get a single string that is not part of an array, it converts it to an array of length 1
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        
        #Tokenizes text
        context = self.text_processor(texts).to_tensor()

        #Calls above call method to get correctly weighted vectors from tokens
        context = self(context)

        return context



