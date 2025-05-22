"""Handles the evaluation of image captioning models.

This module provides functions for generating captions using beam search
and greedy search, evaluating these captions against ground truth using
COCO evaluation metrics, and managing hypothesis objects during beam search.
"""

import json
import os
import time

import tensorflow as tf

from absl import logging
from cocoapi.pycocoevalcap.eval import COCOEvalCap
from dataset import prepare_eval_data

# from models import build_model # Removed as it's unused in run_evaluation
from six.moves import xrange
from training import get_checkpoint_manager
from tqdm import tqdm


class Hypothesis(object):
    """Represents a hypothesis (a candidate sequence) in beam search.

    A hypothesis stores the sequence of tokens generated so far, its cumulative
    log probability, and the last decoder state used to generate the latest token.
    This allows the beam search to continue extending this hypothesis.
    """

    def __init__(self, tokens, log_prob, state):
        """Initializes a new hypothesis.

        Args:
          tokens: A list of integers representing the token IDs generated so far.
          log_prob: The cumulative log probability of this token sequence.
          state: The decoder state after generating the last token in `tokens`.
                 This state is used to predict subsequent tokens.
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def extend(self, token, log_prob, new_state):
        """Creates a new hypothesis by extending the current one with a new token.

        This method is used during beam search when a new token is predicted.
        It creates a new `Hypothesis` object that includes the new token,
        updates the cumulative log probability, and stores the new decoder state.

        Args:
          token: An integer, the ID of the new token to add to the sequence.
          log_prob: The log probability of generating this `token` given the
                    previous state.
          new_state: The decoder state after generating `token`.

        Returns:
          A new `Hypothesis` object representing the extended sequence.
        """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob, new_state)

    @property
    def latest_token(self):
        """Returns the last token in the hypothesis's sequence."""
        return self.tokens[-1]

    def __str__(self):
        """Returns a string representation of the hypothesis."""
        return "Hypothesis(log prob = %.4f, tokens = %s)" % (self.log_prob, self.tokens)


def best_hypothesis(hyps, normalize_by_length):
    """Sorts a list of hypothesis objects based on their log probabilities.

    Hypotheses can optionally be normalized by their token sequence length
    before sorting. The sorting is done in descending order of log probability
    (i.e., best hypothesis first).

    Args:
        hyps: A list of `Hypothesis` objects to be sorted.
        normalize_by_length: A boolean. If True, the log probability of each
            hypothesis is divided by the number of tokens in its sequence
            before comparison. This can favor shorter, high-probability sequences.

    Returns:
        A list of `Hypothesis` objects, sorted in descending order of their
        (optionally normalized) log probabilities.
    """
    # This length normalization is only effective for the final results.
    if normalize_by_length:
        return sorted(hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
    else:
        return sorted(hyps, key=lambda h: h.log_prob, reverse=True)


def generate_captions_with_beam_search(
    model,
    img_features,
    sequence_length,
    vocabulary,
    beam_width=3,
    normalize_by_length=True,
):
    """Generates captions for a batch of images using beam search.

    This function implements the beam search algorithm to generate captions for a
    given batch of image features. At each step of the decoding process, it keeps
    track of the `beam_width` most probable sequences (hypotheses).

    The process involves:
    1. Initializing hypotheses with the start token.
    2. Iteratively extending hypotheses by predicting the next token using the model's
       decoder.
    3. At each step, selecting the `beam_width` hypotheses with the highest
       cumulative log probabilities.
    4. If a hypothesis generates an end token, it's moved to the results.
    5. The search continues until `sequence_length` is reached or `beam_width`
       results are found.
    6. The best hypothesis (highest log probability, optionally normalized by length)
       is chosen for each image in the batch.

    Args:
        model: The trained image captioning model (`models.ImageCaptionModel`),
            containing encoder and decoder components.
        img_features: A tensor of image features, typically with shape
            (batch_size, num_features, feature_dim).
        sequence_length: The maximum length of the captions to be generated.
        vocabulary: A `text.Vocabulary` object used for converting tokens
            to IDs and vice-versa.
        beam_width: An integer specifying the number of hypotheses to keep at
            each step of the beam search.
        normalize_by_length: A boolean. If True, the log probability of each
            completed hypothesis is divided by its length to mitigate favoring
            shorter sequences.

    Returns:
        A list of lists, where each inner list contains the token IDs of the
        generated caption for the corresponding image in the batch.
    """

    # get model components (encoder, decoder and vocabulary)
    encoder = model.encoder
    decoder = model.decoder
    # tokenizer = model.tokenizer # This was assigned but vocabulary.tokenizer was used directly
    start_token = vocabulary.start
    end_token = vocabulary.end

    # get batch size
    batch_size = img_features.shape[0]
    # Initialization of hidden states
    batch_hidden = decoder.reset_state(batch_size=batch_size)
    # Passes visual features through encoder
    batch_features = encoder(img_features)
    predicted_sequences = []
    for idx in tqdm(range(batch_size), desc="beam"):
        # Initialize the hypothesis: start_token will be the initial input
        # Replicate the initial states K times for the first step.
        hyps = [
            Hypothesis([tf.convert_to_tensor(start_token)], 0.0, batch_hidden[idx])
        ] * beam_width
        features = tf.stack([batch_features[idx]] * beam_width)
        # Run beam search
        results = []
        steps = 0
        while steps < sequence_length and len(results) < beam_width:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]

            # Last tokens become next decoder input, a tensor with shape (beam_size, 1)
            dec_input = tf.expand_dims(latest_tokens, 1)
            # Convert array of hidden states to tensor with shape (beam_size, rnn_units)
            hidden = tf.convert_to_tensor(states)
            # Pass input, image features and hidden state to get new predictions (output) and hidden state.
            # Predictions is tensor with shape (beam_size, vocabulary_size)
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            topk_log_probs, topk_ids = tf.nn.top_k(predictions, k=beam_width * 2)

            # Extend each hypothesis.
            all_hyps = []
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            for i in xrange(num_beam_source):
                hyp, hid = hyps[i], hidden[i]
                for j in xrange(beam_width * 2):
                    all_hyps.append(
                        hyp.extend(topk_ids[i, j], topk_log_probs[i, j], hid)
                    )

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for h in best_hypothesis(all_hyps, normalize_by_length):
                if h.latest_token == end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == beam_width or len(results) == beam_width:
                    break
            steps += 1

        if steps == sequence_length:
            results.extend(hyps)

        best_hyp = best_hypothesis(results, normalize_by_length)[0]
        predicted_sequences.append([t.numpy() for t in best_hyp.tokens])

    return predicted_sequences


def generate_captions_with_greedy_search(
    model, img_features, sequence_length, vocabulary
):
    """Generates captions for a batch of image features using greedy search.

    In greedy search, at each step of decoding, the token with the highest
    probability is chosen as the next token in the sequence. This process is
    repeated until `sequence_length` is reached or an end-of-sequence token is
    generated (though this implementation doesn't explicitly handle EOS token
    stopping, relying on `sequence_length`).

    Args:
        model: The trained image captioning model (`models.ImageCaptionModel`),
            containing encoder and decoder components.
        img_features: A tensor of image features, typically with shape
            (batch_size, num_features, feature_dim).
        sequence_length: The maximum length of the captions to be generated.
        vocabulary: A `text.Vocabulary` object used for obtaining the start
            token ID.

    Returns:
        A numpy array of shape (batch_size, sequence_length) containing the
        token IDs of the generated captions.
    """
    encoder = model.encoder
    decoder = model.decoder
    batch_size = img_features.shape[0]
    hidden = decoder.reset_state(batch_size=batch_size)
    # dec_input shape: (batch_size, 1)
    dec_input = tf.expand_dims([vocabulary.start] * batch_size, 1)
    features = encoder(
        img_features, training=False
    )  # Pass visual features through encoder
    predicted_sequences = []
    for t in range(sequence_length):
        # Passing input, features and hidden state through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden, training=False)
        # predictions shape = (batch_size, vocabulary_size)
        predicted_word_idxs = tf.argmax(predictions, axis=1)
        predicted_sequences.append(predicted_word_idxs)
        dec_input = tf.expand_dims(predicted_word_idxs, 1)
    predicted_sequences = tf.stack(predicted_sequences, axis=1)
    return predicted_sequences.numpy()


def generate_captions_for_evaluation(model, eval_dataset, vocabulary, config):
    """Generates captions for a given dataset and saves them to a file.

    This function takes a trained model and an evaluation dataset. It generates
    captions for each image using either beam search or greedy search (based on
    `config.use_beam_search`). The results (image IDs and generated captions)
    are saved in a JSON file specified by `config.eval_result_file`.
    It also handles loading the latest model checkpoint.

    Args:
        model: The `models.ImageCaptionModel` for caption generation.
        eval_dataset: `dataset.DataSet` object with evaluation data.
        vocabulary: `text.Vocabulary` object for token-to-text conversion.
        config: Configuration object with settings like checkpoint directory,
                beam search parameters, and output file paths.

    Returns:
        A list of dictionaries, each with 'image_id' and 'caption'.
    """
    dataset = eval_dataset.dataset
    num_examples = eval_dataset.num_instances
    batch_size = eval_dataset.batch_size
    num_batches = eval_dataset.num_batches
    optimizer = tf.optimizers.get(config.optimizer)  # For checkpoint restoration

    logging.info(f"Evaluating {num_examples} examples.")
    logging.info(f"Divided into {num_batches} batches of size {batch_size}.")

    ckpt_manager, ckpt = get_checkpoint_manager(
        model, optimizer, config.checkpoints_dir, config.max_checkpoints
    )
    if ckpt_manager.latest_checkpoint:
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        # assert_consumed() raises an error because of delayed restoration
        # See https://www.tensorflow.org/alpha/guide/checkpoints#delayed_restorations
        # status.assert_consumed()
        status.assert_existing_objects_matched()
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}")
    else:
        logging.warning(
            "No checkpoint found. Using uninitialized model for evaluation."
        )

    results = []
    current_image_idx = 0
    for _batch_num, (img_features, target) in tqdm(
        enumerate(dataset), desc="Generating Captions", total=num_batches
    ):
        actual_batch_size = target.shape[0]
        sequence_length = target.shape[1]

        if config.use_beam_search:
            predicted_sequences = generate_captions_with_beam_search(
                model,
                img_features,
                sequence_length,
                vocabulary,
                config.beam_width,
                config.normalize_by_length,
            )
        else:
            predicted_sequences = generate_captions_with_greedy_search(
                model, img_features, sequence_length, vocabulary
            )

        for k in range(actual_batch_size):
            sequence = predicted_sequences[k]
            predicted_caption = vocabulary.seq2text(sequence)
            image_id = eval_dataset.image_ids[current_image_idx].item()
            results.append({"image_id": image_id, "caption": predicted_caption})
            current_image_idx += 1

    eval_dir = os.path.abspath(config.eval_result_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    with open(config.eval_result_file, "w") as handle:
        json.dump(results, handle)
    logging.info(f"Evaluation results saved to {config.eval_result_file}")

    return results


def run_evaluation(config):
    """Orchestrates the full evaluation process for image captioning.

    This function performs end-to-end evaluation:
    1. Prepares evaluation dataset (`prepare_eval_data`).
    2. (Assumes captions are already generated by `generate_captions_for_evaluation`
       and results are in `config.eval_result_file`. If not, that function
       should be called here or prior to this.)
    3. Loads generated captions from the results file.
    4. Initializes COCO evaluation API with ground truth and generated captions.
    5. Runs COCO evaluation (BLEU, METEOR, ROUGE, CIDEr scores).
    6. Logs completion and evaluation time.

    Args:
        config: Configuration object with paths (dataset, results file) and settings.
    """
    _eval_dataset, _vocabulary, coco_eval_gt = prepare_eval_data(config)

    # Note: The original code assumes generate_captions_for_evaluation has been run
    # and results are available in config.eval_result_file.
    # If generation is needed:
    # model = build_model(config, vocabulary)
    # generate_captions_for_evaluation(model, eval_dataset, vocabulary, config)

    logging.info(f"Loading ground truth from: {config.eval_captions_file}")
    logging.info(f"Loading generated results from: {config.eval_result_file}")

    if not os.path.exists(config.eval_result_file):
        logging.error(
            f"Evaluation results file not found: {config.eval_result_file}. "
            "Please run caption generation first."
        )
        return

    start_time = time.time()
    coco_eval_result = coco_eval_gt.loadRes(config.eval_result_file)

    scorer = COCOEvalCap(coco_eval_gt, coco_eval_result)
    scorer.evaluate()
    logging.info("Evaluation complete.")
    logging.info("Total evaluation time: %d seconds", time.time() - start)
