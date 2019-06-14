import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3, NASNetLarge
from tensorflow.keras.layers import Dense, Embedding

class ImageCaptionModel(object):
    """CNN-Encoder + RNN-Decoder model with attention for image captioning.

    """

    def __init__(self, num_features, embedding_dim, rnn, rnn_units, weight_initialization, vocabulary, use_attention = True):
        """Creates a new instance ofg ImageCaptionModel class.
        
        Arguments:
            embedding_dim (integer): Number of dimensions of the embedding layer in the RNN_Decoder
            rnn (string): name identifying the type of recurrent units, 'gru' or 'lstm'
            rnn_units (integer): Number of hidden units in the RNN_Decoder
            weight_initialization (string): identifies the type of weight initialization 
            vocabulary (text.Vocabulary): Vocabulary from the training set
        """
        self.encoder = CNN_Encoder(num_features)
        self.decoder = RNN_Decoder(embedding_dim, rnn, rnn_units, vocabulary.size, weight_initialization, use_attention)
        self.tokenizer = vocabulary.tokenizer


class BahdanauAttention(tf.keras.Model):
    """Attention model based on Bahdanau soft attention model.

    """

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, patches, num_features)
        # patches = number of image patches in last conv layer, eg. inception is 8x8 = 64

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, patches, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, patches, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, num_features)
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
    def __init__(self, num_features):
        super(CNN_Encoder, self).__init__()
        self.fc = Dense(num_features)

    def call(self, x):
        # shape of x == (batch_size, patches, channels)
        x = self.fc(x)
        # shape of x after fc == (batch_size, patches, num_features)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    """Decoder model that takes output from encoder and uses RNN to generate captions.

    """

    def __init__(self, embedding_dim, rnn, units, vocab_size, weight_initialization, use_attention):
        super(RNN_Decoder, self).__init__()
        if rnn=='gru':
            from tensorflow.keras.layers import GRU as RNNLayer
        else:
            from tensorflow.keras.layers import LSTM as RNNLayer
        
        self.rnn_type = rnn
        self.units = units
        self.use_attention = use_attention

        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = RNNLayer(units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer=weight_initialization)
        self.fc1 = Dense(units)
        self.fc2 = Dense(vocab_size)
        self.attention = BahdanauAttention(units) if use_attention else None

    def call(self, x, features, hidden):

        # defining attention as a separate model
        # shape of context_vector == (batch_size, num_features)
        # TODO Check: according to some tutorials context vector shape = (batch_size, hidden_size)
        # but I'm getting (batch_size, num_features)
        # shape of attention = (batch_size, patches, 1)
        if self.use_attention:
            context_vector, attention_weights = self.attention(features, hidden)
        else:
            attention_weights = None

        # x shape after passing through embedding == (batch_size, 1, num_features)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, num_features + hidden_size)
        if self.use_attention:
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the RNN
        if self.rnn_type == 'gru':
            output, state = self.rnn(x)
        else: # self.rnn_type =='lstm'
            output, state, _ = self.rnn(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def build_model(config, vocabulary):
    """Builds end-to-end model with CNN encoder, RNN decoder, and Attention mechanism.

    This is a helper method that extracts configuration information from the config object
    
    Arguments:
        config (util.Config): Values for various configuration options.
    
    Returns:
        model.ImageCaptionModel: Full model, including encoder, decoder and attention
    """

    model = ImageCaptionModel(
                config.num_features,
                config.embedding_dim,
                config.rnn,
                config.rnn_units,
                config.weight_initialization,
                vocabulary,
                use_attention= config.use_attention)
    return model
    
