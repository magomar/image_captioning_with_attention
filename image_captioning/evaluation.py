import os

import tensorflow as tf

from absl import logging
from dataset import prepare_eval_data
from models import build_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from training import get_checkpoint_manager

@tf.function
def eval_step(model, img_features, target, loss_function):
    """Forward propagation pass for testing.

    Arguments:
        model (mode.ImageCaptionModel): object containing encoder and decoder
        img_features (tensor): Minibatch of image features, with shape = (batch_size, feature_size, num_features).
            feature_size and num_features depend on the CNN used for the encoder, for example with Inception-V3
            the image features are 8x8x2048, which results in a shape of  (batch_size, 64, 20148).
        target (tensor): Minibatch of tokenized captions, shape = (batch_size, max_captions_length).
            max_captions_length depends on the dataset being used, 
            for example, in COCO 2014 dataset max_captions_length = 53.
        loss_function (tf.losses.Loss): Object that computes the loss function.
            Actually only the SparseCategorialCrossentry is supported
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Object used to tokenize the captions
    
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


def eval(model, val_dataset, coco_eval, vocabulary, config):
    """Evaluates the model on the given dataset
    
    Arguments:
        model {models.ImageCaptionModel} -- The full image captioning model
        val_dataset {dataset.DataSet} -- Evaluation dataset
        config (util.Config): Values for various configuration options
    
    Returns:
        list of float -- List of losses per batch of training
    """
    # Get the evaluation dataset and parameters.
    dataset = val_dataset.dataset
    num_examples = val_dataset.num_instances
    batch_size = val_dataset.batch_size
    num_batches = val_dataset.num_batches
      # Instantiate an optimizer.
    optimizer = tf.optimizers.get(config.optimizer)
    # Instantiate a loss function.
    loss_function = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    logging.info("Evaluating %d examples", num_examples)
    logging.info("Divided into %d batches of size %d", num_batches, batch_size)

    # load model from last checkpoint
    ckpt_manager, ckpt = get_checkpoint_manager(model, optimizer, config.checkpoints_dir, config.max_checkpoints)
    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        # assert_consumed() raises an error because of delayed restoration 
        # See https://www.tensorflow.org/alpha/guide/checkpoints#delayed_restorations
        # status.assert_consumed() 
        status.assert_existing_objects_matched()
    epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    results = []
    # Iterate over the batches of the dataset.
    for (batch, (img_features, target)) in tqdm(enumerate(dataset), desc='batch'):
        batch_loss, t_loss = train_step(model, img_features, target, optimizer, loss_function)
        total_loss += t_loss
    
    return epoch


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
    # if not os.path.exists(config.eval_result_dir):
    #         os.mkdir(config.eval_result_dir)

    val_dataset, vocabulary, coco_eval = prepare_eval_data(config)
    model = build_model(config, vocabulary)
    loss = eval(model, val_dataset, coco_eval, vocabulary, config)
    logging.info ('Final loss after = %.6f', loss)
