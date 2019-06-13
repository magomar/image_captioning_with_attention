import time

import matplotlib.pyplot as plt
import tensorflow as tf

from absl import logging
from dataset import prepare_train_data
from models import build_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.train import Checkpoint, CheckpointManager
from tqdm import tqdm

def compute_loss(labels, predictions, loss_function):
    """Computes loss given labels, predictions and a loss function.
    
    Arguments:
        labels (tensor): ground-truth values
        predictions (tensor): predicted values
        loss_function (tf.keras.losses.Loss): object implementing a loss function, eg. MAE
    
    Returns:
        tensor: computed loss values

    """

    mask = tf.math.logical_not(tf.math.equal(labels, 0))
    loss_ = loss_function(labels, predictions)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def get_checkpoint_manager(model, optimizer, checkpoints_dir, max_checkpoints=None):
    """Obtains a checkpoint manager to manage model saving and restoring.
    
    Arguments:
        model (mode.ImageCaptionModel): object containing encoder, decoder and tokenizer
        optimizer (tf.optimizers.Optimizer): the optimizer used during the backpropagation step
        config (config.Config): Values for various configuration options
    
    Returns:
        tf.train.CheckpointManager, tf.train.Ckeckpoint
    """
    
    ckpt = Checkpoint(encoder = model.encoder,
                      decoder = model.decoder,
                      optimizer = optimizer)
    ckpt_manager = CheckpointManager(ckpt, checkpoints_dir, max_to_keep=max_checkpoints)

    return ckpt_manager, ckpt

@tf.function
def train_step(model, img_features, target, optimizer, loss_function):
    """Forward propagation pass for training.

    Arguments:
        model (mode.ImageCaptionModel): object containing encoder and decoder
        img_features (tensor): Minibatch of image features, with shape = (batch_size, feature_size, num_features).
            feature_size and num_features depend on the CNN used for the encoder, for example with Inception-V3
            the image features are 8x8x2048, which results in a shape of  (batch_size, 64, 20148).
        target (tensor): Minibatch of sequences, shape = (batch_size, sequence_length).
            max_captions_length depends on the dataset being used, 
            for example, in COCO 2014 dataset max_captions_length = 53.
        optimizer (tf.optimizers.Optimizer): the optimizer used during the backpropagation step.
        loss_function (tf.losses.Loss): Object that computes the loss function.
            Actually only the SparseCategorialCrossentry is supported
    
    Returns:
        loss: loss value for one step (a mini-batch )
        total_loss: loss value averaged by the size of captions
    """

    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    loss = 0

    # Obtain the actual size of this batch, since it may differ from predefined batchsize
    # when running the last batch of an epoch
    batch_size=target.shape[0]
    sequence_length=target.shape[1]

    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=batch_size)

    # Expands input to decoder
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

    # Open a GradientTape to record the operations run during the forward pass, 
    # which enables autodifferentiation.
    with tf.GradientTape() as tape:
        # Passes visual features through encoder
        features = encoder(img_features)
        for i in range(1, sequence_length):
            # Passing input, features and hidden state through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += compute_loss(target[:, i], predictions, loss_function)
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(sequence_length))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def fit(model, train_dataset, config):
    """Fits the model to the provided dataset
    
    Arguments:
        model {models.ImageCaptionModel} -- The full image captioning model
        train_dataset {dataset.DataSet} -- Training dataset
        config (util.Config): Values for various configuration options
    
    Returns:
        list of float -- List of losses per batch of training
    """

    # Get the training dataset and various parameters.
    dataset = train_dataset.dataset
    num_examples = train_dataset.num_instances
    batch_size = train_dataset.batch_size
    num_batches = train_dataset.num_batches
    num_epochs = config.num_epochs
    # Instantiate an optimizer.
    optimizer = tf.optimizers.get(config.optimizer)
    # Instantiate a loss function.
    loss_function = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    logging.info("Training on %d examples for %d epochs", num_examples, num_epochs)
    logging.info("Divided into %d batches of size %d", num_batches, batch_size)

    # Try to resume training from saved checkpoints
    resume_training = False
    if config.resume_from_checkpoint:
        ckpt_manager, ckpt = get_checkpoint_manager(model, optimizer, config.checkpoints_dir, config.max_checkpoints)
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            # assert_consumed() raises an error because of delayed restoration 
            # See https://www.tensorflow.org/alpha/guide/checkpoints#delayed_restorations
            # status.assert_consumed() 
            status.assert_existing_objects_matched()
            resume_training = True
    
    if resume_training:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        if start_epoch < config.num_epochs:
            logging.info("Resuming training from epoch %d", start_epoch) 
    else:
        start_epoch = 0
        logging.info("Starting training from scratch")

    batch_losses = []

    # Iterate over epochs.
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        total_loss = 0
        print("*** Epoch %d ***" % epoch)
        # Iterate over the batches of the dataset.
        for (batch, (img_features, target)) in tqdm(enumerate(dataset), desc='batch', total=num_batches):
            batch_loss, t_loss = train_step(model, img_features, target, optimizer, loss_function)
            total_loss += t_loss

            # if batch % 100 == 0:
            #     caption_length = int(target.shape[1])
            #     logging.info('Epoch %d Batch %d/%d Loss: %.4f',
            #         epoch + 1, batch, num_batches, batch_loss.numpy() / caption_length)
        # Storing the epoch end loss value to plot later
        batch_losses.append(total_loss / num_batches)

        logging.info ('Epoch %d Loss %.6f', epoch + 1, total_loss / num_batches)
        logging.info ('Time taken for 1 epoch: %d sec\n', time.time() - start)

        # Save checkpoint for the last epoch
        ckpt_manager.save()

    return batch_losses

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

def train(config):
    """Orchestrates the training process.
    
    This method is responsible of executing all the steps required to train a new model, 
    which includes:
    - Preparing the dataset
    - Building the model
    - Fitting the model to the data

    Arguments:
        config (util.Config): Values for various configuration options
    """
    logging.info("cnn = %s", config.cnn)
    logging.info("rnn = %s", config.rnn)
    logging.info("embedding_dim = %s", config.embedding_dim)
    logging.info("rnn_units = %s", config.rnn_units)
    logging.info("num_features = %s", config.num_features)
    logging.info("weight_initialization = %s", config.weight_initialization)
    logging.info("batch_size = %s", config.batch_size)
    logging.info("optimizer = %s", config.optimizer)

    train_dataset, vocabulary = prepare_train_data(config)
    model = build_model(config, vocabulary)
    start = time.time()
    losses = fit(model, train_dataset, config)
    if len(losses) > 0:
        logging.info('Total training time: %d seconds', time.time() - start)
        logging.info('Final loss after %d epochs = %.6f', config.num_epochs, losses[-1])
        print('Final loss after %d epochs = %.6f' % (config.num_epochs, losses[-1]))
        plot_loss(losses)
    else:
        print("No training done, since number of epochs was reached")
