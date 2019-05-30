import json
import os

import numpy as np
import tensorflow as tf

from absl import logging
from cocoapi.pycocotools.coco import COCO
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text import load_or_build_vocabulary, Vocabulary
from tqdm import tqdm


class DataSet(object):
    """Combines images and captions into a single Dataset object.

    This class builds a tf.data.Dataset with pairs of image files and captions,
    ready for batch processing. It also includes the tokenizer object.
    """
    
    def __init__(self,
                name,
                image_ids,
                image_files,
                captions,
                batch_size,
                shuffle=False,
                buffer_size=1000,
                drop_remainder=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.captions = captions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder
        self.setup()

    def setup(self):
        """Setup the dataset.

        """

        self.num_instances = len(self.image_files)
        self.num_batches = int(np.ceil(self.num_instances * 1.0 / self.batch_size))
        # self.num_batches = self.num_instances // self.batch_size

        dataset = Dataset.from_tensor_slices((self.image_files, self.captions))
        # using map to load the numpy files in parallel
        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                map_image_features_to_caption, [item1, item2], [tf.float32, tf.int32]
                ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        # shuffling and batching the train dataset
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=self.drop_remainder)
        else:
            dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)

        self.dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def map_image_features_to_caption(image_file, caption):
    """ Load image features from npy file and maps them to caption.
    
    """

    image_features = np.load(image_file.decode('utf-8')+'.npy')
    return image_features, caption

def prepare_train_data(config):
    """Prepare the data for training the model.
    
    Arguments:
        config (util.Config): Values for various configuration options.
    
    Returns:
        dataset.DataSet: Training dataset in tensorflow format, ready for batch consumption
    """

    logging.info("Preparing training data for %s...", config.dataset_name)

    # obtaining the image ids, image files and text captions
    coco = COCO(config.train_captions_file)
    image_ids = coco.get_all_image_ids()
    image_files = coco.get_image_files(config.train_image_dir, image_ids)
    text_captions = coco.get_all_captions()

    logging.info("Number of instances in the training set: %d", len(image_ids))

    num_examples = config.num_train_examples
    if num_examples is not None:
        # selecting the first num_examples
        logging.info("Using just %d instances for training", num_examples)
        image_ids = image_ids[:num_examples]
        image_files = image_files[:num_examples]
        text_captions = text_captions[:num_examples]
    else:
        logging.info("Using full training dataset")

    # vocabulary = load_or_build_vocabulary(config, sentences = text_captions)
    vocabulary = load_or_build_vocabulary(config)

    captions = vocabulary.process_sentences(text_captions)

    dataset = DataSet(
        '%s_%s'.format(config.dataset_name, 'Training'),
        image_ids,
        image_files,
        captions,
        config.batch_size,
        shuffle= True,
        buffer_size= config.buffer_size,
        drop_remainder= config.drop_remainder
    )

    return dataset, vocabulary


def prepare_eval_data(config):
    """ Prepare the data for evaluating the model.
    
    Arguments:
        config (util.Config): Values for various configuration options.
    
    Returns:
        tf.data.Dataset: Evaluation dataset in tensorflow format, ready for batch consumption
    """
    
    logging.info("Preparing validation data for %s...", config.dataset_name)

    # obtaining the image ids, image files and text captions
    coco = COCO(config.eval_captions_file)
    image_ids = coco.get_unique_image_ids()
    dataset_size = len(image_ids)
    logging.info("Total number of instances in the validation set: %d", dataset_size)
    if config.filter_by_caption_length:
        coco.filter_by_cap_len(config.max_caption_length)
    image_ids = coco.get_unique_image_ids()

    if len(image_ids) < dataset_size:
        # selecting the first num_examples
        logging.info("Using just %d images for the evaluation phase", len(image_ids))
    else:
        logging.info("Using full validation set")

    image_files = coco.get_image_files(config.eval_image_dir, image_ids)
    text_captions = coco.get_example_captions(image_ids)

    vocabulary = load_or_build_vocabulary(config)

    captions = vocabulary.process_sentences(text_captions)

    dataset = DataSet(
        '%s_%s'.format(config.dataset_name, 'Validation'),
        image_ids,
        image_files,
        captions,
        config.batch_size,
        shuffle= False,
        buffer_size= config.buffer_size,
        drop_remainder= False
    )

    return dataset, vocabulary, coco

def download_coco(config):
    """Donwload COCO dataset.
    
    """

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