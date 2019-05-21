import json
import os

import numpy as np
import tensorflow as tf

from absl import logging
from models import get_image_features_extract_model
from tqdm import tqdm
from util import ImageHelper, shuffle_lists
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
                tokenizer,
                batch_size,
                shuffle=False,
                buffer_size=1000,
                drop_remainder=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.captions = captions
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # TODO: replace by self.shuffle = True when shuffling is fixed in TF2
        # see https://github.com/tensorflow/tensorflow/issues/28552
        self.shuffle = False
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
    

def get_raw_data(image_dir, caption_file, image_prefix):
    image_ids = []
    raw_captions = []

    # read the json file with the raw captions
    with open(os.path.abspath(caption_file), 'r') as f:
        annotations = json.load(f)

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        img_id = annot['image_id']
        image_ids.append(img_id)
        raw_captions.append(caption)

    image_helper = ImageHelper(image_dir, image_prefix)
    image_files = [image_helper.get_image_file(img_id) for img_id in image_ids]

    return image_ids, image_files, raw_captions

def map_image_features_to_caption(image_file, caption):
    """ Load image features from npy file and maps them to caption """

    image_features = np.load(image_file.decode('utf-8')+'.npy')
    return image_features, caption

def preprocess_images(config):
    """Extract image features and save them as numpy arrays.
    
    This will process both the training and the validation datasets.
    """

    logging.info("Preprocessing images...")

    batchsize = config.image_features_batchsize
    # TODO save image features in specific folder (config.features_dir)
    # features_path = os.path.abspath(config.image_features_dir)

    # Obtain image files for training dataset
    train_image_ids, train_image_files, train_raw_captions = get_raw_data(
        config.train_image_dir, config.train_caption_file, config.train_image_prefix)

    # Obtain image files for evaluation dataset
    eval_image_ids, eval_image_files, eval_raw_captions = get_raw_data(
        config.eval_image_dir, config.eval_caption_file, config.eval_image_prefix)

    # Create feature extraction layer
    pretrained_image_model = get_image_features_extract_model(config.cnn)

    # Create a dataset with images ready to be fed into the encoder in batches 
    encode_set = sorted(set(train_image_files + eval_image_files))
    image_dataset = Dataset.from_tensor_slices(encode_set)
    if config.cnn == 'inception_v3':
        image_dataset = image_dataset.map(
            load_image_inception_v3, num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).batch(batchsize)
    elif config.cnn == 'nasnet':
        image_dataset = image_dataset.map(
            load_image_nasnet, num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).batch(batchsize)

    logging.info("Extracting image features and saving them to disk")

    for image, image_file in tqdm(image_dataset):
        batch_features = pretrained_image_model(image)
        batch_features = tf.reshape(
            batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        # Save image features with same name of image plus
        for bf, p in zip(batch_features, image_file):
            img_path = p.numpy().decode("utf-8")
            np.save(img_path, bf.numpy())


def prepare_train_data(config):
    """Prepare the data for training the model.
    
    Args:
        config (util.Config): Values for various configuration options.
    
    Returns:
        tf.data.Dataset: Training dataset in tensorflow format, ready for batch consumption
    """

    logging.info("Preparing training data for %s...", config.dataset_name)

    # obtaining the raw captions and the image files
    __, image_files, raw_captions = get_raw_data(
        config.train_image_dir,config.train_image_dir, config.train_image_prefix)

    logging.info("Number of instances in the training set: %d", len(image_files))

    # selecting the first num_examples from the shuffled sets
    num_examples = config.num_examples
    if num_examples is not None:
        logging.info("Using just %d instances for training", num_examples)
        # perhaps shuffling the captions and image_names together, setting a random state
        image_files, raw_captions = shuffle_lists(image_files, raw_captions)
        image_files = image_files[:num_examples]
        raw_captions = raw_captions[:num_examples]
    else:
        logging.info("Using full training dataset")
    
    captions, tokenizer = preprocess_captions(raw_captions, config.max_vocabulary_size)

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

    ## obtaining the raw captions and the image files
    __, image_files, raw_captions = get_raw_data(
        config.eval_image_dir,config.eval_image_dir, config.eval_image_prefix)

    logging.info("Number of instances in the evaluation set: %d", len(image_files))

    # selecting the first num_examples from the shuffled sets
    num_examples = config.num_examples
    if num_examples is not None:
        logging.info("Using just %d instances for training", num_examples)
        # perhaps shuffling the captions and image_names together, setting a random state
        image_files, raw_captions = shuffle_lists(image_files, raw_captions)
        image_files = image_files[:num_examples]
        raw_captions = raw_captions[:num_examples]
    else:
        logging.info("Using full training dataset")
    
    captions, tokenizer = preprocess_captions(raw_captions, config.max_vocabulary_size)

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


def get_tokenizer(captions, vocabulary_size):
    
    # choosing the top k words from the vocabulary
    tokenizer = Tokenizer(num_words=vocabulary_size,
                            oov_token="<unk>",
                            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    # adds a padding token
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    return tokenizer

def preprocess_captions(captions, max_vocabulary_size):
    """Tokenize and pad captions, restricted to max vocabulary size.
    
    Args:
        captions (list): A list of sentences
        max_vocabulary_size (integer): Max vocabulary size
    
    Returns:
        captions: tokenized & padded captions
        tokenizer: the tokenizer configured for the given captions
    """

    logging.info("Preprocessing captions...")

    # obtains a tokenizer to process captions, limits the vocabulary size
    tokenizer = get_tokenizer(captions, max_vocabulary_size)

    vocab_size = len(tokenizer.word_index) + 1
    logging.info("Full vocabulary size: %d", vocab_size)
    logging.info("Vocabulary size restricted to %d words", max_vocabulary_size)

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
    captions = pad_sequences(
        sequences, maxlen=max_length, padding='post')
    return captions, tokenizer

def load_image_inception_v3(image_file):
    """Loads an image from file, and transforms it into the Inception-V3 format.
    
    Image data should be reshaped to (299, 299, 3)

    Arguments:
        image_file {String} -- Path to image file
    
    Returns:
        tensor -- Image features with shape (8, 8, 2048)
    """

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, image_file

def load_image_nasnet(image_file):
    """Loads an image from file, and transforms it into the NASNet format.
    
    Image data should be reshaped to (299, 299, 3)

    Arguments:
        image_file {String} -- Path to image file
    
    Returns:
        tensor -- Image features with shape (, 4032)
    """
    

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.nasnet.preprocess_input(image)
    return image, image_file

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