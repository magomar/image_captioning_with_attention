import os
import matplotlib
import random
# matplotlib.use('agg')

import matplotlib.pyplot as plt
import tensorflow as tf

class ImageHelper(object):
    def __init__(self, image_path, prefix):
        self.path = os.path.abspath(image_path)
        self.prefix = prefix

    def load_image(self, image_id):
        image = tf.io.read_file(get_image_file(image_id))
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def get_image_file(self, image_id):
        image_file = os.path.join(self.path, '%s%012d.jpg' % (self.prefix, image_id))
        return image_file

def generate_caption_argmax(model, img_features, sequence_length):
    # get model components (encoder, decoder and tokenizer)
    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=1)
    # Expands input to decoder
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
        dec_input = tf.expand_dims([predicted_word_idx], 0)
    return " ".join(predicted_caption)


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
        actual_batch_size=target.shape[0]
        sequence_length=target.shape[1]

        batch_results = []
        for k, (img_feat, tgt) in enumerate(zip(img_features, target)):
            predicted_caption = generate_caption_argmax(model, img_feat, sequence_length) 
            batch_results.append({'image_id': eval_dataset.image_ids[i].item(),
                                  'caption': predicted_caption,
                                #   'true caption': vocabulary.sequence2sentence(tgt.numpy())
                                  'ground_truth': vocabulary.sequence2sentence(eval_dataset.captions[i])
                                  })
            i += 1
        results.extend(batch_results)
    
    # Save results to file
    eval_dir = os.path.abspath(config.eval_result_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    with open(config.eval_result_file, 'w') as handle:
        json.dump(results, handle)

    return results

    ????????????????

    def generate_sequences_beam(model, img_features, sequence_length, beam_size=3):
    # get model components (encoder, decoder and tokenizer)
    encoder = model.encoder
    decoder = model.decoder
    tokenizer = model.tokenizer
    # get batch size and caption length
    actual_batch_size=target.shape[0]
    sequence_length=target.shape[1]
    # Initializing the hidden state for each batch, since captions are not related from image to image
    hidden = decoder.reset_state(batch_size=actual_batch_size)
    # Expands input to decoder, inserts a dimesion of 1 at axis 1
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * actual_batch_size, 1)
    # Passes visual features through encoder
    features = encoder(img_features)

    contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

    partial_caption_data = []
    complete_caption_data = []
    for k in range(config.batch_size):
        initial_beam = CaptionData(sequence = [],
                                    memory = initial_memory[k],
                                    output = initial_output[k],
                                    score = 1.0)
        partial_caption_data.append(TopN(config.beam_size))
        partial_caption_data[-1].push(initial_beam)
        complete_caption_data.append(TopN(config.beam_size))

    return results
