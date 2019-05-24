import json
import os

import tensorflow as tf

from absl import logging
from dataset import prepare_eval_data
from models import build_model
from training import get_checkpoint_manager
from tqdm import tqdm

def generate_captions(model, eval_dataset, vocabulary, config):
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
        actual_batch_size=target.shape[0]
        sequence_length=target.shape[1]

        batch_results = []
        for k, (img_feat, tgt) in enumerate(zip(img_features, target)):
            predicted_caption = generate_caption(model, img_feat, sequence_length) 
            batch_results.append({'image_id': eval_dataset.image_ids[i].item(),
                                  'caption': predicted_caption,
                                #   'true caption': vocabulary.sequence2sentence(tgt.numpy())
                                  'true caption': vocabulary.sequence2sentence(eval_dataset.captions[i])
                                  })
            i += 1
        results.extend(batch_results)
    
    return results

def generate_caption(model, img_features, sequence_length):
    # get model components (encoder, decoder and tokenizer)
    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=1)
    # Expands input to decoder, inserts a dimesion of 1 at axis 1
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    # Passes visual features through encoder
    features = encoder(img_features)
    predicted_caption = []
    for i in range(sequence_length):
        # Passing input, features and hidden state through the decoder
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_word_idx = tf.argmax(predictions[0]).numpy()
        predicted_word = tokenizer.index_word[predicted_word_idx]
        if predicted_word == '<end>':
            break
        else:
            predicted_caption.append(predicted_word)
    return "".join(predicted_caption)


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
    eval_dir = os.path.abspath(config.eval_result_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    eval_dataset, vocabulary, coco_eval = prepare_eval_data(config)
    model = build_model(config, vocabulary)
    results = generate_captions(model, eval_dataset, vocabulary, config)
    with open(config.eval_result_file, 'w') as handle:
        json.dump(results, handle)
