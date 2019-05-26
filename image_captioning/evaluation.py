import json
import os
import time

import tensorflow as tf

from absl import logging
from cocoapi.pycocoevalcap.eval import COCOEvalCap
from dataset import prepare_eval_data
from models import build_model
from training import get_checkpoint_manager
from tqdm import tqdm

class CaptionData(object):
    def __init__(self, sequence, memory, output, score):
       self.sequence = sequence
       self.memory = memory
       self.output = output
       self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score

class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []


def generate_sequences_argmax(model, img_features, sequence_length):
    """Generate captions for a batch of image features
    
    Arguments:
        model {models.ImageCaptionModel} -- The full image captioning model
        img_features {tensor} -- Image features, shape = (batch_size, 64, 2048)
        sequence_length {integer} -- length of captions
    
    Returns:
        list of generated captions
    """
    # get model components (encoder, decoder and tokenizer)
    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    # get batch size and caption length
    batch_size=img_features.shape[0]
    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=batch_size)
    # Expands input to decoder, generates a batch of sequences of length=1, with 
    # word index corresponding to '<start>', that is, shape = (64,1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)
    # Passes visual features through encoder
    features = encoder(img_features)
    predicted_sequences = []
    for i in range(sequence_length):
        # Passing input, features and hidden state through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        # predictions shape = (batch_size, vocabulary_size)
        predicted_word_idxs = tf.argmax(predictions, axis=1)
        predicted_sequences.append(predicted_word_idxs)
        dec_input = tf.expand_dims(predicted_word_idxs, 1)
    predicted_sequences = tf.stack(predicted_sequences,axis=1)
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
        # caption_data = generate_captions_argmax(model, img_features, vocabulary)
        predicted_sequences = generate_sequences_argmax(model, img_features, sequence_length)
        for k, sequence in enumerate(predicted_sequences):
            # sequence = caption_data[k][0].sequence
            # score = caption_data[k][0].score
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
    # start = time.time()
    # results = eval(model, eval_dataset, vocabulary, config)
    # logging.info('Total caption generation time: %d seconds', time.time() - start)

    # Evaluate these captions
    start = time.time()
    coco_eval_result = coco_eval.loadRes(config.eval_result_file)
    scorer = COCOEvalCap(coco_eval, coco_eval_result)
    scorer.evaluate()
    logging.info("Evaluation complete.")
    logging.info('Total evaluation time: %d seconds', time.time() - start)