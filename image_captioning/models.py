import tensorflow as tf

from tensorflow.keras.layers import Dense, Embedding, GRU

class ImageCaptionModel(object):
    """CNN-Encoder + RNN-Decoder model with attention for image captioning.

    """

    def __init__(self, embedding_dim, rnn_units, tokenizer):
        """Creates a new instance ofg ImageCaptionModel class.
        
        Args:
            embedding_dim (integer): Number of dimensions of the embedding layer in the RNN_Decoder
            rnn_units (integer): Number of hidden units in the RNN_Decoder
            tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer fitted to the training captions
        """
        self.encoder = CNN_Encoder(embedding_dim)
        self.vocab_size = len(tokenizer.word_index) + 1
        self.decoder = RNN_Decoder(embedding_dim, rnn_units, self.vocab_size)
        self.tokenizer = tokenizer

    def compile(optimizer, loss):
        """Configures the model for training.
        
        Args:
            optimizer (String): name of optimizer (or optimizer instance)
            loss (String): name of objective function (or objective function)
        Raises:
            ValueError: In case of invalid arguments for `optimizer`, `loss`
        """
        self.optimizer = optimizers.get


class BahdanauAttention(tf.keras.Model):
    """Attention model based on Bahdanau soft attention model.

    """

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    """Encoder model to process the image features.

    This encoder assumes images are pretrained using a CNN. 
    That is, this model only has to add the fully connected layer
    Instead of images it will receive as inputs the image features from the pretrained model
    """

    # We have already extracted the features and saved them as npy arrays
    # This encoder only has to pass those features through a fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    """Decoder model that takes output from encoder and uses RNN to generate captions.

    """

    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = Dense(self.units)
        self.fc2 = Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):

        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def build_model(tokenizer, config):
    """Builds end-to-end model with CNN encoder, RNN decoder, and Attention mechanism.

    This is a helper method that extracts configuration information from the config object
    
    Args:
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer fitted to the training captions
        config (util.Config): Values for various configuration options.
    
    Returns:
        model.ImageCaptionModel: Full model, including encoder, decoder and tokenizer
    """

    model = ImageCaptionModel(config.embedding_dim, config.rnn_units, tokenizer)
    return model
