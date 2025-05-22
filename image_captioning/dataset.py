"""Handles dataset preparation for image captioning.

This module provides functionalities to load, preprocess, and structure
the COCO dataset for training and evaluation of image captioning models.
It includes the `DataSet` class to manage image features and captions
as a `tf.data.Dataset`, and functions to prepare data for different phases
(training, evaluation) and download the dataset if necessary.
"""

import os

import numpy as np
import tensorflow as tf

from absl import logging
from cocoapi.pycocotools.coco import COCO
from tensorflow.data import Dataset
from text import load_or_build_vocabulary


class DataSet(object):
    """Combines images and captions into a single Dataset object.

    This class builds a tf.data.Dataset with pairs of image files and captions,
    ready for batch processing. It also includes the tokenizer object.
    """

    def __init__(
        self,
        name,
        image_ids,
        image_feature_files,
        captions,
        batch_size,
        shuffle=False,
        buffer_size=1000,
        drop_remainder=False,
    ):
        self.image_ids = np.array(image_ids)
        self.image_features_files = np.array(image_feature_files)
        self.captions = captions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder
        self.setup()

    def setup(self):
        """Configures the tf.data.Dataset object.

        This method calculates the number of instances and batches required for
        the dataset. It then creates a `tf.data.Dataset` from the image
        feature files and captions. The process involves:
        1. Mapping a function to load image features (numpy files) and pair
           them with their corresponding captions. This is done in parallel
           using `tf.data.experimental.AUTOTUNE`.
        2. Shuffling the dataset (if `self.shuffle` is True).
        3. Batching the data according to `self.batch_size`.
        4. Prefetching batches for optimized performance using
           `tf.data.experimental.AUTOTUNE`.
        """

        self.num_instances = len(self.image_features_files)
        self.num_batches = int(np.ceil(self.num_instances * 1.0 / self.batch_size))

        dataset = Dataset.from_tensor_slices((self.image_features_files, self.captions))
        # Using map to load the numpy files in parallel
        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                map_image_features_to_caption, [item1, item2], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # shuffling and batching the train dataset
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size).batch(
                self.batch_size, drop_remainder=self.drop_remainder
            )
        else:
            dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)

        self.dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def map_image_features_to_caption(npy_file, caption):
    """Loads an image feature map from a .npy file and pairs it with a caption.

    This function is designed to be used with `tf.numpy_function` within a
    `tf.data.Dataset.map()` operation. It decodes the file path, loads the
    numpy array containing image features, and returns the features along with
    the corresponding caption.

    Args:
        npy_file: A tf.string tensor representing the path to the .npy file
                  containing the image features.
        caption: A tf.int32 tensor representing the tokenized caption.

    Returns:
        A tuple (image_features, caption) where:
            image_features: A tf.float32 tensor of the loaded image features.
            caption: The input tf.int32 tensor representing the caption.
    """

    image_features = np.load(npy_file.decode("utf-8"))
    return image_features, caption


def prepare_train_data(config):
    """Prepares the training dataset.

    This function orchestrates the preparation of the training data. It involves:
    1. Initializing a COCO API object using the training captions file.
    2. Retrieving image IDs and corresponding filenames.
    3. Constructing paths to the pre-extracted image feature files.
    4. Loading or building the vocabulary from the training captions.
    5. Processing the text captions into tokenized sequences.
    6. Creating a `DataSet` object with the prepared image features, captions,
       and training-specific configurations (batch size, shuffling, etc.).

    Args:
        config: A configuration object containing paths, hyperparameters,
                and settings for dataset preparation.

    Returns:
        A tuple (dataset, vocabulary) where:
            dataset: A `DataSet` object ready for training, containing batched
                     image features and captions.
            vocabulary: The `Vocabulary` object used for processing captions.
    """

    logging.info("Preparing training data for %s...", config.dataset_name)

    # obtaining the image ids, image files and text captions
    coco = COCO(config.train_captions_file)
    image_ids = coco.get_all_image_ids()
    dataset_size = len(image_ids)
    logging.info("Total number of instances in the training set: %d", dataset_size)
    image_ids = coco.get_all_image_ids()

    if len(image_ids) < dataset_size:
        logging.info("Using just %d images for the training phase", len(image_ids))
    else:
        logging.info("Using full validation set")

    image_filenames = coco.get_image_filenames(image_ids)
    image_features_files = [
        os.path.join(config.image_features_dir, config.cnn, f"{filename}.npy")
        for filename in image_filenames
    ]
    text_captions = coco.get_all_captions()

    vocabulary = load_or_build_vocabulary(config, text_captions)

    captions = vocabulary.process_sentences(text_captions)

    dataset = DataSet(
        f"{config.dataset_name}_Training",
        image_ids,
        image_features_files,
        captions,
        config.batch_size,
        shuffle=True,
        buffer_size=config.buffer_size,
        drop_remainder=config.drop_remainder,
    )

    return dataset, vocabulary


def prepare_eval_data(config):
    """Prepares the evaluation dataset.

    Similar to `prepare_train_data`, this function prepares the dataset for
    evaluation. Key differences include:
    - It uses the evaluation captions file (`config.eval_captions_file`).
    - It retrieves unique image IDs to avoid redundant evaluations.
    - It typically does not shuffle the data.
    - It also returns the COCO API object for evaluation metrics calculation.

    Args:
        config: A configuration object containing paths, hyperparameters,
                and settings for dataset preparation.

    Returns:
        A tuple (dataset, vocabulary, coco) where:
            dataset: A `DataSet` object ready for evaluation.
            vocabulary: The `Vocabulary` object used for processing captions.
            coco: The COCO API object initialized with evaluation annotations,
                  useful for official evaluation script.
    """

    logging.info("Preparing validation data for %s...", config.dataset_name)

    # obtaining the image ids, image files and text captions
    coco = COCO(config.eval_captions_file)
    image_ids = coco.get_unique_image_ids()
    dataset_size = len(image_ids)
    logging.info("Total number of instances in the validation set: %d", dataset_size)
    image_ids = coco.get_unique_image_ids()

    if len(image_ids) < dataset_size:
        logging.info("Using just %d images for the evaluation phase", len(image_ids))
    else:
        logging.info("Using full validation set")

    image_filenames = coco.get_image_filenames(image_ids)
    image_features_files = [
        os.path.join(config.image_features_dir, config.cnn, f"{filename}.npy")
        for filename in image_filenames
    ]
    text_captions = coco.get_example_captions(image_ids)

    vocabulary = load_or_build_vocabulary(config)

    captions = vocabulary.process_sentences(text_captions)

    dataset = DataSet(
        f"{config.dataset_name}_Validation",
        image_ids,
        image_features_files,
        captions,
        config.batch_size,
        shuffle=False,
        buffer_size=config.buffer_size,
        drop_remainder=False,
    )

    return dataset, vocabulary, coco


def download_coco(config):
    """Downloads the COCO dataset (captions and images) if not already present.

    This function checks for the existence of COCO training and validation
    data (both captions and images) in the specified directories. If any part
    is missing, it downloads the corresponding zip file from the official COCO
    dataset URLs and extracts it to the appropriate location.

    Args:
        config: A configuration object containing paths for storing the COCO
                dataset (e.g., `config.train_caption_dir`,
                `config.train_image_dir`, `config.eval_image_dir`).
    """

    if not os.path.exists(config.train_caption_dir):
        # captions_zip variable is unused
        tf.keras.utils.get_file(
            "captions.zip",  # Name for the downloaded file
            cache_subdir=os.path.abspath("./data/coco"),
            origin="http://images.cocodataset.org/captions/captions_trainval2014.zip",
            extract=True,
        )
    if not os.path.exists(config.train_image_dir):
        # image_zip variable is unused
        tf.keras.utils.get_file(
            "train2014.zip",  # Name for the downloaded file, matching origin
            cache_subdir=os.path.abspath("./data/coco"),
            origin="http://images.cocodataset.org/zips/train2014.zip",
            extract=True,
        )
    if not os.path.exists(config.eval_image_dir):
        # image_zip variable is unused
        tf.keras.utils.get_file(
            "val2014.zip",  # Name for the downloaded file, matching origin
            cache_subdir=os.path.abspath("./data/coco"),
            origin="http://images.cocodataset.org/zips/val2014.zip",
            extract=True,
        )
