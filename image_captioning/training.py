"""Handles the training process for the image captioning model.

This module includes functions for:
- Computing the loss between predicted and actual captions.
- Managing model checkpoints (saving and restoring).
- Executing individual training steps (forward and backward pass).
- Orchestrating the overall training loop (`fit` function).
- Plotting training loss.
- A main `train` function to coordinate the training phase.
"""

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
    """Computes the loss, ignoring padded parts of the sequences.

    Calculates loss between true labels and predictions, masking padding tokens
    (label == 0) so they don't contribute to the loss.

    Args:
        labels: Ground-truth token IDs. Shape: (batch_size, sequence_length).
        predictions: Model's predicted logits.
                     Shape: (batch_size, sequence_length, vocab_size) or
                     (batch_size, vocab_size) if used per time-step.
        loss_function: Keras loss object (e.g., SparseCategoricalCrossentropy)
                       initialized with `reduction='none'`.

    Returns:
        A scalar tensor: mean loss over unmasked tokens.
    """
    mask = tf.math.logical_not(tf.math.equal(labels, 0))  # Mask out padding
    loss_ = loss_function(labels, predictions)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_checkpoint_manager(model, optimizer, checkpoints_dir, max_checkpoints=None):
    """Creates and returns a TensorFlow CheckpointManager and Checkpoint.

    Used for saving and restoring model weights and optimizer state.

    Args:
        model: The `ImageCaptionModel` instance (must have `encoder` and
               `decoder` attributes).
        optimizer: The `tf.keras.optimizers.Optimizer` used for training.
        checkpoints_dir: Directory to save checkpoints.
        max_checkpoints: Max number of recent checkpoints to keep.
                         Defaults to None (keeps all).

    Returns:
        A tuple (tf.train.CheckpointManager, tf.train.Checkpoint).
    """
    ckpt = Checkpoint(encoder=model.encoder, decoder=model.decoder, optimizer=optimizer)
    ckpt_manager = CheckpointManager(ckpt, checkpoints_dir, max_to_keep=max_checkpoints)

    return ckpt_manager, ckpt


@tf.function
def train_step(model, img_features, target, optimizer, loss_function):
    """Performs a single training step (forward and backward pass).

    Executes model's forward pass, calculates loss, computes gradients, and
    applies them. Uses "teacher forcing" by feeding ground-truth target tokens
    as decoder inputs.

    Args:
        model: The `ImageCaptionModel` instance.
        img_features: Batch of image features.
                      Shape: (batch_size, num_patches, feature_depth).
        target: Batch of ground-truth caption sequences (token IDs).
                Shape: (batch_size, sequence_length).
        optimizer: `tf.keras.optimizers.Optimizer` for applying gradients.
        loss_function: `tf.keras.losses.Loss` (e.g., SparseCategoricalCrossentropy).

    Returns:
        A tuple (batch_loss, total_batch_loss_avg), where:
            batch_loss: Total loss for the batch (summed over time steps).
            total_batch_loss_avg: Average loss per sequence in the batch.
    """
    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    loss = 0

    # Obtain the actual size of this batch, since it may differ from predefined batchsize
    # when running the last batch of an epoch
    batch_size = target.shape[0]
    sequence_length = target.shape[1]

    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=batch_size)

    # Expands input to decoder
    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]] * batch_size, 1)

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

    total_loss = loss / int(sequence_length)
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def fit(model, train_dataset, config):
    """Trains the image captioning model for a specified number of epochs.

    Orchestrates the main training loop: iterates over epochs and batches,
    calling `train_step` for each batch to update model weights and accumulate loss.

    Key features:
    - Checkpoint Management: Saves model/optimizer states. Resumes from the
      latest checkpoint if `config.resume_from_checkpoint` is True. Checkpoints
      are saved based on `config.checkpoints_frequency`.
    - Epoch Iteration: Trains for `config.num_epochs`.
    - Loss Tracking: Calculates and logs average loss per epoch.
    - Optimizer/Loss: Uses optimizer from `config` and
      `SparseCategoricalCrossentropy` (within `train_step`).

    Args:
        model: The `ImageCaptionModel` to train.
        train_dataset: `dataset.DataSet` with training data (batched image
                       features and target captions).
        config: `Config` object with training parameters (epochs, batch size,
                optimizer, checkpoint settings, etc.).

    Returns:
        A list of floats: average loss per epoch. Empty if no epochs run.
    """
    dataset = train_dataset.dataset
    num_examples = train_dataset.num_instances
    batch_size = train_dataset.batch_size
    num_batches = train_dataset.num_batches
    num_epochs = config.num_epochs
    # Instantiate an optimizer.
    optimizer = tf.optimizers.get(config.optimizer)
    # Instantiate a loss function.
    loss_function = SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    logging.info("Training on %d examples for %d epochs", num_examples, num_epochs)
    logging.info("Divided into %d batches of size %d", num_batches, batch_size)

    # Try to resume training from saved checkpoints
    resume_training = False
    if config.resume_from_checkpoint:
        ckpt_manager, ckpt = get_checkpoint_manager(
            model, optimizer, config.checkpoints_dir, config.max_checkpoints
        )
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            # assert_consumed() raises an error because of delayed restoration
            # See https://www.tensorflow.org/alpha/guide/checkpoints#delayed_restorations
            # status.assert_consumed()
            status.assert_existing_objects_matched()
            resume_training = True

    if resume_training:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
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
        for batch, (img_features, target) in tqdm(
            enumerate(dataset), desc="batch", total=num_batches
        ):
            batch_loss, t_loss = train_step(
                model, img_features, target, optimizer, loss_function
            )
            total_loss += t_loss

            # if batch % 100 == 0:
            #     caption_length = int(target.shape[1])
            #     logging.info('Epoch %d Batch %d/%d Loss: %.4f',
            #         epoch + 1, batch, num_batches, batch_loss.numpy() / caption_length)
        # Storing the epoch end loss value to plot later
        batch_losses.append(total_loss / num_batches)

        logging.info("Epoch %d Loss %.6f", epoch + 1, total_loss / num_batches)
        logging.info("Time taken for 1 epoch: %d sec\n", time.time() - start)

        # Save checkpoint for the last epoch (with certain frequency)
        if epoch % config.checkpoints_frequency == 0:
            ckpt_manager.save()

    return batch_losses


def plot_loss(losses):
    """Plots the training loss over epochs.

    Args:
        losses (list of float): A list where each element is the average loss
                                for an epoch.
    """
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.show()


def train(config):
    """Orchestrates the entire training phase.

    Sets up the training environment from `config`:
    1. Logs configuration parameters.
    2. Prepares training dataset and vocabulary (`prepare_train_data`).
    3. Builds the model (`build_model`).
    4. Trains the model using `fit`.
    5. Logs training time and final loss.
    6. Optionally plots loss (commented out).

    Args:
        config: `Config` object with all settings for dataset preparation,
                model building, and training.
    """
    logging.info("cnn = %s", config.cnn)
    logging.info("rnn = %s", config.rnn)
    logging.info("embedding_dim = %s", config.embedding_dim)
    logging.info("rnn_units = %s", config.rnn_units)
    logging.info("num_features = %s", config.num_features)
    logging.info("use_attention = %s", config.use_attention)
    logging.info("weight_initialization = %s", config.weight_initialization)
    logging.info("batch_size = %s", config.batch_size)
    logging.info("optimizer = %s", config.optimizer)
    logging.info("dropout = %s", config.dropout)

    train_dataset, vocabulary = prepare_train_data(config)
    model = build_model(config, vocabulary)
    start = time.time()
    losses = fit(model, train_dataset, config)
    if len(losses) > 0:
        logging.info("Total training time: %d seconds", time.time() - start)
        logging.info("Final loss after %d epochs = %.6f", config.num_epochs, losses[-1])
        print("Final loss after %d epochs = %.6f" % (config.num_epochs, losses[-1]))
        # plot_loss(losses)
    else:
        print("No training done, since number of epochs was reached")
