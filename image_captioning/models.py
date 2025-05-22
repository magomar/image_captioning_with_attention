"""Defines the neural network architectures for image captioning.

This module contains the definitions for:
- `ImageCaptionModel`: The main model class that combines the encoder and decoder.
- `CNN_Encoder`: An encoder model to process image features from a pretrained CNN.
- `BahdanauAttention`: An attention mechanism based on Bahdanau's soft attention.
- `RNN_Decoder`: A recurrent neural network (RNN) decoder that generates captions.
It also includes a helper function `build_model` to instantiate the full model.
"""

import tensorflow as tf

# from tensorflow.keras import Model # Removed as tf.keras.Model is used directly
from tensorflow.keras.layers import Dense, Embedding, Dropout


class ImageCaptionModel(object):
    """CNN-Encoder + RNN-Decoder model with attention for image captioning."""

    def __init__(
        self,
        num_features,
        embedding_dim,
        rnn,
        rnn_units,
        weight_initialization,
        dropout,
        vocabulary,
        use_attention=True,
    ):
        """Creates an ImageCaptionModel instance.

        Encapsulates `CNN_Encoder` and `RNN_Decoder`.

        Args:
            num_features: Output features from `CNN_Encoder`'s dense layer.
            embedding_dim: Dimensionality of token embeddings in `RNN_Decoder`.
            rnn: RNN type for the decoder ('gru' or 'lstm').
            rnn_units: Number of units in the decoder's RNN layer.
            weight_initialization: Keras initializer for RNN weights.
            dropout: Dropout rate for encoder and decoder.
            vocabulary: `text.Vocabulary` object with tokenizer and word maps.
            use_attention: If True, `RNN_Decoder` uses `BahdanauAttention`.
        """
        self.encoder = CNN_Encoder(num_features, dropout)
        self.decoder = RNN_Decoder(
            embedding_dim,
            rnn,
            rnn_units,
            vocabulary.size,
            weight_initialization,
            dropout,
            use_attention,
        )
        self.tokenizer = vocabulary.tokenizer


class CNN_Encoder(tf.keras.Model):
    """Processes pre-extracted image features using a fully connected layer.

    Takes image features (e.g., from a pre-trained CNN's last conv layer)
    and passes them through a Dense layer with ReLU activation. Dropout is
    applied during training.

    Input shape: (batch_size, num_image_patches, feature_channels).
    Output shape: (batch_size, num_image_patches, num_features).
    """

    def __init__(self, num_features, dropout):
        """Initializes CNN_Encoder.

        Args:
            num_features: Dimensionality of the output feature space (Dense layer units).
            dropout: Dropout rate for training.
        """
        super(CNN_Encoder, self).__init__()
        self.do = Dropout(dropout)
        self.fc = Dense(num_features)

    def call(self, x, training=True):
        """Forward pass of the encoder.

        Args:
            x (tf.Tensor): Input tensor of image features.
                           Shape: (batch_size, num_patches, feature_channels).
            training (bool): Whether the model is in training mode (for dropout).

        Returns:
            tf.Tensor: Processed image features.
                       Shape: (batch_size, num_patches, num_features).
        """
        # x shape: (batch_size, patches, channels)
        if training:
            x = self.do(x)
        x = self.fc(x)
        # x shape after fc: (batch_size, patches, num_features)
        x = tf.nn.relu(x)
        return x


class BahdanauAttention(tf.keras.Model):
    """Implements Bahdanau-style (additive) soft attention.

    Calculates attention weights over encoder outputs based on the current
    decoder hidden state. These weights produce a context vector (a weighted
    sum of encoder features) for the decoder.
    """

    def __init__(self, units, dropout):
        """Initializes BahdanauAttention.

        Args:
            units: Number of units in Dense layers (W1, W2, V), often related
                   to decoder's RNN units.
            dropout: Dropout rate for score calculation during training.
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.do = Dropout(dropout)
        self.V = Dense(1)

    def call(self, features, hidden, training=True):
        """Computes attention weights and context vector.

        Args:
            features (tf.Tensor): Output from the CNN encoder.
                                  Shape: (batch_size, num_patches, num_features).
            hidden (tf.Tensor): Hidden state from the decoder's RNN.
                                Shape: (batch_size, rnn_units).
            training (bool): Whether the model is in training mode (for dropout).

        Returns:
            tuple:
                - context_vector (tf.Tensor): Weighted sum of encoder features.
                                            Shape: (batch_size, num_features).
                - attention_weights (tf.Tensor): Attention weights over encoder features.
                                               Shape: (batch_size, num_patches, 1).
        """
        # features shape: (batch_size, patches, num_features)
        # hidden shape: (batch_size, hidden_size)
        # hidden_with_time_axis shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape: (batch_size, patches, units)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        if training:
            score = self.do(score)  # Apply dropout to the score
        # attention_weights shape: (batch_size, patches, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape before sum: (batch_size, patches, num_features)
        context_vector = attention_weights * features
        # context_vector shape after sum: (batch_size, num_features)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    """Generates captions token by token using RNN and optional attention.

    Takes encoded image features and the previously generated token to predict
    the next token. Components:
    - Embedding layer for input tokens.
    - RNN layer (GRU or LSTM).
    - Dense layer for vocabulary scores.
    - Optional BahdanauAttention for context vector.
    """

    def __init__(
        self,
        embedding_dim,
        rnn,
        units,
        vocab_size,
        weight_initialization,
        dropout,
        use_attention,
    ):
        """Initializes RNN_Decoder.

        Args:
            embedding_dim: Dimensionality of the token embedding layer.
            rnn: RNN type ('gru' or 'lstm').
            units: Number of units in the RNN layer.
            vocab_size: Size of the vocabulary for the output layer.
            weight_initialization: Keras initializer for RNN recurrent weights.
            dropout: Dropout rate for the final classification layer.
            use_attention: If True, use BahdanauAttention.
        """
        super(RNN_Decoder, self).__init__()
        if rnn == "gru":
            from tensorflow.keras.layers import GRU as RNNLayer
        else:
            from tensorflow.keras.layers import LSTM as RNNLayer

        self.rnn_type = rnn
        self.units = units
        self.use_attention = use_attention

        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = RNNLayer(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer=weight_initialization,
        )
        self.fc1 = Dense(units)
        self.do = Dropout(dropout)
        self.fc2 = Dense(vocab_size)
        self.attention = BahdanauAttention(units, dropout) if use_attention else None

    def call(self, x, features, hidden, training=True):
        """Forward pass of the decoder for one time step.

        Args:
            x (tf.Tensor): Input token IDs for the current time step.
                           Shape: (batch_size, 1).
            features (tf.Tensor): Output from the CNN encoder.
                                  Shape: (batch_size, num_patches, num_features).
            hidden (tf.Tensor): Hidden state from the previous time step.
                                Shape: (batch_size, rnn_units).
            training (bool): Whether the model is in training mode (for dropout
                             in attention and final layer).

        Returns:
            tuple:
                - predictions (tf.Tensor): Logits over the vocabulary.
                                           Shape: (batch_size, vocab_size).
                - state (tf.Tensor): New hidden state from the RNN.
                                     Shape: (batch_size, rnn_units).
                - attention_weights (tf.Tensor or None): Attention weights if attention is used,
                                                      else None. Shape: (batch_size, num_patches, 1).
        """
        if self.use_attention:
            context_vector, attention_weights = self.attention(
                features, hidden, training
            )
        else:
            # If not using attention, sum features across patches to get a fixed context
            context_vector = tf.reduce_sum(features, axis=1)
            attention_weights = None

        # x shape after embedding: (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation: (batch_size, 1, embedding_dim + num_features)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passing the concatenated vector to the RNN
        if self.rnn_type == "gru":
            output, state = self.rnn(x)  # output shape: (batch_size, 1, units)
        else:  # self.rnn_type == 'lstm'
            output, state, _ = self.rnn(x)  # output shape: (batch_size, 1, units)

        # Pass RNN output through a Dense layer
        # x shape: (batch_size, 1, units)
        x = self.fc1(output)
        if training:
            x = self.do(x)  # Apply dropout

        # Reshape to (batch_size, units) before final classification layer
        x = tf.reshape(x, (-1, x.shape[2]))

        # Final classification layer to get vocabulary scores
        # x shape: (batch_size, vocab_size)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        """Resets the initial hidden state of the RNN to zeros."""
        return tf.zeros((batch_size, self.units))


def build_model(config, vocabulary):
    """Builds and returns the complete ImageCaptionModel.

    Instantiates `ImageCaptionModel` using parameters from the `config` object
    and `vocabulary`.

    Args:
        config: `Config` object with model hyperparameters (e.g., `num_features`,
                `embedding_dim`, `rnn_units`).
        vocabulary: `text.Vocabulary` object with tokenizer and vocab size.

    Returns:
        An instance of `ImageCaptionModel`.
    """

    model = ImageCaptionModel(
        config.num_features,
        config.embedding_dim,
        config.rnn,
        config.rnn_units,
        config.weight_initialization,
        config.dropout,
        vocabulary,
        use_attention=config.use_attention,
    )
    return model
