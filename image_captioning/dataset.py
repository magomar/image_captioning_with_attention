import os
import json

import tensorflow as tf

from absl import logging

def prepare_train_data(config):
    """ Prepare the data for training the model. """
    logging.info("Preparing training data...")

    # set the path to the captions file
    annotation_file  = os.path.abspath(config.train_caption_file) 

   # set the path to the training images
    train_path = os.path.abspath(config.train_image_dir)

    # read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_captions = []
    all_img_names = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = train_path + '/COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_names.append(full_coco_image_path)
        all_captions.append(caption)

    logging.info("Number of instances in the training set: %d", len(all_captions))

    # shuffling the captions and image_names together, setting a random state
    # all_captions, all_img_names = shuffle(all_captions, all_img_names, random_state=1)

    # selecting the first num_examples from the shuffled sets
    num_examples = config.num_examples
    if num_examples is not None:
        captions = all_captions[:num_examples]
        img_names = all_img_names[:num_examples]
        logging.info("Using just %d instances for training", len(captions))
    else:
        logging.info("Using full training dataset", len(captions))
    
    # for i in range(5):
    #     img = tf.io.read_file(img_names[i])
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     plt.imshow(img)
    #     plt.show()
    #     print(captions[i])

    captions = preprocess_captions(captions, config.max_vocabulary_size)

    return img_names, captions


def prepare_eval_data(config):
    """ Prepare the data for evaluating the model. """


def get_tokenizer(captions, vocabulary_size):
    # choosing the top k words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocabulary_size,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    # adds a padding token
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    return tokenizer

def preprocess_captions(captions, max_vocabulary_size):
    '''Preprocess captions: Donwload COCO dataset'''

    logging.info("Preprocessing captions...")

    # obtains a tokenizer to process captions, limits the vocabulary size
    tokenizer = get_tokenizer(captions, max_vocabulary_size)

    vocab_size = len(tokenizer.word_index) + 1
    logging.info("Vocabulary size restricted to %d words", max_vocabulary_size)
    logging.info("Actual vocabulary size: %d %d", max_vocabulary_size, vocab_size)

    # converts captions to sequences of tokens
    logging.info("Tokenizing captions...")
    sequences = tokenizer.texts_to_sequences(captions)
    logging.info("Tokenized caption example: %s -> %s", captions[0], sequences[0])

    # compute max length of a sequence
    max_length = max(len(seq) for seq in sequences)
    logging.info("Max caption length: %d", max_length)

    # pad sequences to met the max length
    # note: if maxlen parameter is not provided, pad_sequences calculates that automatically
    logging.info("Padding captions...")
    captions = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post')
    return captions

def download_coco(config):
    '''Donwload COCO dataset'''
    if not os.path.exists(config.train_caption_dir):
        captions_zip = tf.keras.utils.get_file('captions.zip',
                                        cache_subdir=os.path.abspath('./data/coco'),
                                        origin = 'http://images.cocodataset.org/captions/captions_trainval2014.zip',
                                        extract = True)
    if not os.path.exists(config.train_image_dir):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                    cache_subdir=os.path.abspath('./data/coco'),
                                    origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                    extract = True)
    if not os.path.exists(config.eval_image_dir):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                    cache_subdir=os.path.abspath('./data/coco'),
                                    origin = 'http://images.cocodataset.org/zips/val2014.zip',
                                    extract = True)