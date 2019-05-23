import os

import tensorflow as tf
import numpy as np

from dataset import prepare_eval_data, prepare_train_data
from images import load_image_inception_v3

@tf.function
def test_step(model, img_features, target, loss_function):
    """Forward propagation pass for testing.

    Args:
        model (mode.ImageCaptionModel): object containing encoder, decoder and tokenizer
        img_features (tensor): Minibatch of image features, with shape = (batch_size, feature_size, num_features).
            feature_size and num_features depend on the CNN used for the encoder, for example with Inception-V3
            the image features are 8x8x2048, which results in a shape of  (batch_size, 64, 20148).
        target (tensor): Minibatch of tokenized captions, shape = (batch_size, max_captions_length).
            max_captions_length depends on the dataset being used, 
            for example, in COCO 2014 dataset max_captions_length = 53.
        loss_function (tf.losses.Loss): Object that computes the loss function.
            Actually only the SparseCategorialCrossentry is supported
    
    Returns:
        loss: loss value for all the 
        total_loss: loss value averaged by the size of captions ()
    """

    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    loss = 0

    # Obtain the actual size of this batch, since it may differ from predefined batchsize
    # when running the last batch of an epoch
    actual_batch_size=target.shape[0]
    sequence_length=target.shape[1]

    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=actual_batch_size)

    # Expands input to decoder, inserts a dimesion of 1 at axis 1
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * actual_batch_size, 1)
    results = []

    # Passes visual features through encoder
    features = encoder(img_features)
    for i in range(1, sequence_length):

        # Passing input, features and hidden state through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        predicted_word_idxs = tf.argmax(predictions).numpy()

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(sequence_length))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def evaluate(config):
    """Orchestrates the evaluation process.
    
    This method is responsible of executing all the steps required to train a new model, 
    which includes:
    - Preparing the dataset
    - Building the model
    - Fitting the model to the data

    Arguments:
        config (util.Config): Values for various configuration options
    """

    if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

    val_dataset = prepare_val_data(config)
    train_dataset = prepare_train_data(config)
    tokenizer = train_dataset.tokenizer

    imf = val_dataset.image_files[0]
    image, image_file = load_image_inception_v3(imf)
    print(image)
    print(image_file)
