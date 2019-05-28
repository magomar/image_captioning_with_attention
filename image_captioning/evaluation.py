import json
import os
import time

import tensorflow as tf

from absl import logging
from cocoapi.pycocoevalcap.eval import COCOEvalCap
from dataset import prepare_eval_data
from models import build_model
from six.moves import xrange
from training import get_checkpoint_manager
from tqdm import tqdm


class Hypothesis(object):
  """Defines a hypothesis during beam search.
  
  """

  def __init__(self, tokens, log_prob, state):
    """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens, usually 1.
      state: decoder initial states.
    """
    self.tokens = tokens
    self.log_prob = log_prob
    self.state = state

  def extend(self, token, log_prob, new_state):
    """Extend the hypothesis with result from latest step.

    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
    return Hypothesis(self.tokens + [token], 
                      self.log_prob + log_prob,
                      new_state)

  @property
  def latest_token(self):
    return self.tokens[-1]

  def __str__(self):
    return ('Hypothesis(log prob = %.4f, tokens = %s)' %
            (self.log_prob, self.tokens))


def best_hypothesis(hyps, normalize_by_length):
    """Sort the hyps based on log probs and length.

    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    # This length normalization is only effective for the final results.
    if normalize_by_length:
      return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)
    else:
      return sorted(hyps, key=lambda h: h.log_prob, reverse=True)

def generate_captions_with_beam_search(model, img_features, sequence_length, vocabulary, beam_width=3, normalize_by_length=True):
    """Generate captions for a batch of images
    
    Arguments:
        model {models.ImageCaptionModel} -- The full image captioning model
        img_features {tensor} -- Image features, shape = (batch_size, 64, 2048)
        sequence_length {integer} -- length of captions
        vocabulary {text.Vocabulary} -- Vocabulary used to tokenize captions
        beam_width {integer} -- The number of hypothesis kept after each search step
        normalize_by_length {boolean} -- Whether to normalize logits by length
    
    Returns:
        list of generated captions
    """

    # get model components (encoder, decoder and tokenizer)
    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    start_token = vocabulary.start
    end_token = vocabulary.end

    # get batch size 
    batch_size=img_features.shape[0]
    # Initialization of hidden states and decoder inputs
    initial_states = decoder.reset_state(batch_size=batch_size)
    # Passes visual features through encoder
    batch_features = encoder(img_features)
    predicted_sequences = []
    for idx in range(batch_size):
        # Initialize the hypothesis
        # Replicate the initial states K times for the first step.
        hyps = [Hypothesis(tf.convert_to_tensor([start_token]), 0.0, initial_states[idx])] * beam_width
        results = []
        features = tf.stack([batch_features[idx]] * beam_width)
        # Run beam search
        steps = 0
        while steps < sequence_length and len(results) < beam_width:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]

            # Passing input, features and hidden state through the decoder
            dec_input = tf.expand_dims(latest_tokens,1)
            hidden = tf.convert_to_tensor(states)
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            # topk_ids = tf.argsort(predictions, axis=1)[:,-beam_width:]
            # topk_log_probs = predictions[:,topk_ids]
            topk_log_probs, topk_ids = tf.nn.top_k(predictions,k=beam_width*2)

            # topk_log_probs = topk_log_probs.numpy()
            # topk_ids = topk_ids.numpy()
            # hidden = hidden.numpy()

            # Extend each hypothesis.
            all_hyps = []
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            for i in xrange(num_beam_source):
                h, ns = hyps[i], hidden[i]
                for j in xrange(beam_width*2):
                    all_hyps.append(h.extend(topk_ids[i, j], topk_log_probs[i, j], ns))

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
        # predicted_sequences.append(best_hyp.tokens[1:])

    return predicted_sequences


def generate_captions_with_greedy_search(model, img_features, sequence_length, vocabulary):
    """Generate captions for a batch of image features
    
    Arguments:
        model {models.ImageCaptionModel} -- The full image captioning model
        img_features {tensor} -- Image features, shape = (batch_size, 64, 2048)
        sequence_length {integer} -- length of captions
        vocabulary {text.Vocabulary} -. vocabulary used to tokenize the captions
    
    Returns:
        list of generated captions
    """
    # get model components (encoder, decoder and tokenizer)
    encoder = model.encoder
    decoder = model.decoder
    # get batch size and caption length
    batch_size=img_features.shape[0]
    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=batch_size)
    # Expands input to decoder, generates a batch of sequences of length=1, with 
    # word index corresponding to '<start>', that is, shape = (64,1)
    dec_input = tf.expand_dims([vocabulary.start] * batch_size, 1)
    # Passes visual features through encoder
    features = encoder(img_features)
    predicted_sequences = []
    for t in range(sequence_length):
        # Passing input, features and hidden state through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        # predictions shape = (batch_size, vocabulary_size)
        predicted_word_idxs = tf.argmax(predictions, axis=1)
        predicted_sequences.append(predicted_word_idxs)
        dec_input = tf.expand_dims(predicted_word_idxs, 1)
    predicted_sequences = tf.stack(predicted_sequences, axis=1)
    return predicted_sequences.numpy()

def eval(model, eval_dataset, vocabulary, config):
    """Generate captions on the given dataset
    
    Arguments:
        model {models.ImageCaptionModel} -- The full image captioning model
        eval_dataset {dataset.DataSet} -- Evaluation dataset
        config (util.Config): Values for various configuration options
    
    Returns:
        list of float -- List of losses per batch of training
    """

    # Get the evaluation dataset and parameters.
    dataset = eval_dataset.dataset
    num_examples = eval_dataset.num_instances
    batch_size = eval_dataset.batch_size
    num_batches = eval_dataset.num_batches
    # Need an optimizer to recreate model from checkpoint
    # Although it is not being used for evaluation
    optimizer = tf.optimizers.get(config.optimizer)
    
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
    i = 0
    # Iterate over the batches of the dataset.
    for (batch, (img_features, target)) in tqdm(enumerate(dataset), desc='batch'):      
        # Obtain the actual size of this batch,  since it may differ 
        # from predefined batchsize when running the last batch of an epoch
        batch_size=target.shape[0]
        sequence_length=target.shape[1]
        if config.use_beam_search:
            predicted_captions = generate_captions_with_beam_search(
                    model, img_features, sequence_length, vocabulary,
                    config.beam_width, config.normalize_by_length)
        else:
            predicted_captions = generate_captions_with_greedy_search(
                    model, img_features, sequence_length, vocabulary)
        
        for k, sequence in enumerate(predicted_captions):
            predicted_caption = vocabulary.sequence2sentence(sequence)
            results.append({'image_id': eval_dataset.image_ids[i].item(),
                            'caption': predicted_caption
                            # 'ground_truth': vocabulary.sequence2sentence(eval_dataset.captions[i])
                            })
            i += 1

    # Save results to file
    eval_dir = os.path.abspath(config.eval_result_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    with open(config.eval_result_file, 'w') as handle:
        json.dump(results, handle)

    return results

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

    eval_dataset, vocabulary, coco_eval = prepare_eval_data(config)
    model = build_model(config, vocabulary)
    start = time.time()
    results = eval(model, eval_dataset, vocabulary, config)
    logging.info('Total caption generation time: %d seconds', time.time() - start)

    # Evaluate these captions
    start = time.time()
    coco_eval_result = coco_eval.loadRes(config.eval_result_file)
    scorer = COCOEvalCap(coco_eval, coco_eval_result)
    scorer.evaluate()
    logging.info("Evaluation complete.")
    logging.info('Total evaluation time: %d seconds', time.time() - start)