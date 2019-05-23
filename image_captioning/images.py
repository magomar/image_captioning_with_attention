import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import logging
from cocoapi.pycocotools.coco import COCO
from tqdm import tqdm

def preprocess_images(config):
    """Extract image features and save them as numpy arrays.
    
    This will process both the training and the validation datasets.
    
    Arguments:
        config (util.Config): Values for various configuration options.
    """

    logging.info("Preprocessing images (extracting image features)...")

    batchsize = config.image_features_batchsize
    # TODO save image features in specific folder (config.features_dir)
    # features_path = os.path.abspath(config.image_features_dir)

    coco = COCO(config.train_captions_file)
    
    # Obtain image files for training dataset
    train_image_ids, train_image_files, train_text_captions = get_raw_data(
        config.train_image_dir, config.train_caption_file, config.train_image_prefix)

    # Obtain image files for evaluation dataset
    eval_image_ids, eval_image_files, eval_text_captions = get_raw_data(
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

def plot_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    plt.imshow(image)
    plt.show()