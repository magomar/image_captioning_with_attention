import json
import os

import numpy as np
import tensorflow as tf

from absl import logging
from images import preprocess_images
from models import get_image_features_extract_model
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from util import ImageHelper, shuffle_lists
from tqdm import tqdm
from text import build_vocabulary, Vocabulary


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
        # self.num_batches = int(np.ceil(self.num_instances * 1.0 / self.batch_size))
        self.num_batches = self.num_instances // self.batch_size

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
    

# def get_raw_data(image_dir, captions_annot_file, image_prefix):
#     """Get raw data to build a dataset.
    
#     Arguments:
#         image_dir {String} -- Relative path of folder with the images
#         captions_annot_file {String} -- Relative path of JSON file with the captions
#         image_prefix {String} -- Prefix used to name image files (prepended to image identifiers)
    
#     Returns:
#         A list of image identifiers, image files and raw captions
#     """

#     image_ids = []
#     text_captions = []

#     # read the json file with the raw captions
#     with open(os.path.abspath(captions_annot_file), 'r') as f:
#         annotations = json.load(f)

#     for annot in annotations['annotations']:
#         caption = '<start> ' + annot['caption'] + ' <end>'
#         img_id = annot['image_id']
#         image_ids.append(img_id)
#         text_captions.append(caption)

#     image_helper = ImageHelper(image_dir, image_prefix)
#     image_files = [image_helper.get_image_file(img_id) for img_id in image_ids]

#     return image_ids, image_files, text_captions

def map_image_features_to_caption(image_file, caption):
    """ Load image features from npy file and maps them to caption.
    
    """

    image_features = np.load(image_file.decode('utf-8')+'.npy')
    return image_features, caption

def prepare_train_data(config):
    """Prepare the data for training the model.
    
    Args:
        config (util.Config): Values for various configuration options.
    
    Returns:
        dataset.DataSet: Training dataset in tensorflow format, ready for batch consumption
    """

    logging.info("Preparing training data for %s...", config.dataset_name)

    # obtaining the image ids, image files and text captions
    image_ids, image_files, text_captions = get_raw_data(
        config.train_image_dir, config.train_captions_annot_file, config.train_image_prefix)

    logging.info("Number of instances in the training set: %d", len(image_files))

    # selecting the first num_examples from the shuffled sets
    num_examples = config.num_examples
    if num_examples is not None:
        logging.info("Using just %d instances for training", num_examples)
        # perhaps shuffling the captions and image_names together, setting a random state
        image_files, text_captions = shuffle_lists(image_files, text_captions)
        image_files = image_files[:num_examples]
        text_captions = text_captions[:num_examples]
    else:
        logging.info("Using full training dataset")
    
    captions, tokenizer = preprocess_captions(text_captions, config.max_vocabulary_size)

    dataset = DataSet(
        '%s_%s'.format(config.dataset_name, 'Training'),
        image_ids,
        image_files,
        captions,
        tokenizer,
        config.batch_size,
        shuffle= True,
        buffer_size= config.buffer_size,
        drop_remainder= config.drop_remainder
    )

    return dataset


def prepare_eval_data(config):
    """ Prepare the data for evaluating the model.
    
    Args:
        config (util.Config): Values for various configuration options.
    
    Returns:
        tf.data.Dataset: Evaluation dataset in tensorflow format, ready for batch consumption
    """
    
    logging.info("Preparing validation data for %s...", config.dataset_name)

   # obtaining the image ids, image files and text captions
    image_ids, image_files, text_captions = get_raw_data(
        config.eval_image_dir, config.eval_captions_annot_file, config.eval_image_prefix)

    logging.info("Number of instances in the evaluation set: %d", len(image_files))
    
    captions, tokenizer = preprocess_captions(text_captions, config.max_vocabulary_size)

    dataset = DataSet(
        '%s_%s'.format(config.dataset_name, 'Training'),
        image_ids,
        image_files,
        captions,
        tokenizer,
        config.batch_size,
        shuffle= False,
        buffer_size= config.buffer_size,
        drop_remainder= config.drop_remainder
    )

    return dataset

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