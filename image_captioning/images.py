"""Manages image loading, preprocessing, and feature extraction.

This module provides functions to:
- Preprocess images for input into various CNN architectures.
- Extract image features using pretrained CNN models (e.g., InceptionV3, VGG16).
- Define image size constants for different models.
- Offer utility functions for image handling, like plotting.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import logging
from cocoapi.pycocotools.coco import COCO
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tqdm import tqdm

# IMAGE_SIZE: A dictionary mapping CNN model names to their expected input
# image dimensions (height, width). This is used to resize images
# appropriately before feeding them into a specific pretrained model.
IMAGE_SIZE = {
    "vgg16": (224, 224),
    "inception_v3": (299, 299),
    "xception": (299, 299),
    "resnet50": (224, 224),
    "nasnet_large": (331, 331),
    "inception_resnet_v2": (299, 299),
}


def preprocess_images(config):
    """Extracts image features from training and validation datasets and saves them.

    This function performs the following steps:
    1. Initializes a pre-trained CNN model specified by `config.cnn` for
       feature extraction.
    2. Gathers unique image file paths from both training
       (`config.train_captions_file`) and validation
       (`config.eval_captions_file`) sets.
    3. Creates a `tf.data.Dataset` from these image files.
    4. Maps a preprocessing function (`image_preprocessing_function`) to load,
       decode, resize, and preprocess each image according to the chosen CNN's
       requirements.
    5. Batches the preprocessed images.
    6. Iterates through the batches, passes them through the CNN encoder to get
       image features.
    7. Reshapes the features to (batch_size, num_features, feature_depth).
    8. Saves each image's feature map as a separate .npy file in the directory
       specified by `config.image_features_dir/config.cnn/`.

    The primary purpose is to pre-calculate and cache image features to speed up
    the training and evaluation of image captioning models.

    Args:
        config: A configuration object containing paths to datasets, CNN model
                choice, batch size for feature extraction, and directory for
                saving features.
    """
    logging.info("Preprocessing images (extracting image features)...")

    cnn = config.cnn

    batchsize = config.image_features_batchsize

    # If needed create folder for saving image features
    features_dir = config.image_features_dir
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # Create feature extraction layer
    encoder = get_image_encoder(cnn)

    coco_train = COCO(config.train_captions_file)
    train_image_ids = coco_train.get_unique_image_ids()

    coco_eval = COCO(config.eval_captions_file)
    eval_image_ids = coco_eval.get_unique_image_ids()

    image_files = coco_train.get_image_files(
        config.train_image_dir, train_image_ids
    ) + coco_eval.get_image_files(config.eval_image_dir, eval_image_ids)

    # Create a dataset with images ready to be fed into the encoder
    # (CNN for extracting image features) in batches.
    image_dataset = Dataset.from_tensor_slices(sorted(set(image_files)))
    image_dataset = (
        image_dataset.map(
            image_preprocessing_function(cnn),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(batchsize)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )  # Added prefetch here for consistency with dataset.py

    features_cnn_dir = os.path.join(features_dir, cnn)
    if not os.path.exists(features_cnn_dir):
        os.makedirs(features_cnn_dir)

    logging.info(f"Extracting features using {cnn} and saving to {features_cnn_dir}")
    for images, image_filenames in tqdm(image_dataset, desc="Extracting Features"):
        image_features_batch = encoder(images)
        image_features_batch = tf.reshape(
            image_features_batch,
            (
                image_features_batch.shape[0],
                -1,
                image_features_batch.shape[3],
            ),
        )
        for img_features, img_filename_tensor in zip(
            image_features_batch, image_filenames
        ):
            img_filename_str = img_filename_tensor.numpy().decode("utf-8")
            features_path = os.path.join(
                features_cnn_dir, os.path.basename(img_filename_str)
            )  # Save features with .npy extension implicitly by np.save
            np.save(features_path, img_features.numpy())


def image_preprocessing_function(cnn):
    """Returns a callable for loading and preprocessing images for a specific CNN.

    This higher-order function takes a CNN model name and returns a tailored
    `load_and_preprocess_image` function. This returned function handles:
    1. Reading the image file.
    2. Decoding JPEG images.
    3. Resizing to the CNN's required input size (from `IMAGE_SIZE`).
    4. Applying model-specific preprocessing (Keras applications' `preprocess_input`).

    Args:
        cnn: Name of the CNN model (e.g., 'vgg16', 'inception_v3'). Used to
             get image size and the correct `preprocess_input` function.

    Returns:
        A function that takes an image file path, returning a preprocessed image
        tensor and the original image file path.
    """
    image_size = IMAGE_SIZE[cnn]
    if cnn == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif cnn == "inception_v3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif cnn == "xception":
        from tensorflow.keras.applications.xception import preprocess_input
    elif cnn == "nasnet_large":
        from tensorflow.keras.applications.nasnet import preprocess_input
    elif cnn == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif cnn == "inception_resnet_v2":
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

    def load_and_preprocess_image(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = preprocess_input(image)
        return image, image_file

    return load_and_preprocess_image


def get_image_encoder(cnn):
    """Creates and returns a pre-trained CNN model for feature extraction.

    Loads a Keras Application model (e.g., VGG16, InceptionV3) with ImageNet
    weights, excluding the top classification layer. The model is configured to
    output features from its last convolutional (or equivalent) layer.

    Args:
        cnn: Name of the CNN model. Supported: 'vgg16', 'inception_v3',
             'xception', 'nasnet_large', 'resnet50', 'inception_resnet_v2'.

    Returns:
        A `tf.keras.Model` for image feature extraction.
    """
    if cnn == "vgg16":
        from tensorflow.keras.applications.vgg16 import VGG16 as PTModel
    elif cnn == "inception_v3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3 as PTModel
    elif cnn == "xception":
        from tensorflow.keras.applications.xception import Xception as PTModel
    elif cnn == "nasnet_large":
        from tensorflow.keras.applications.nasnet import NASNetLarge as PTModel
    elif cnn == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50 as PTModel
    elif cnn == "inception_resnet_v2":
        from tensorflow.keras.applications.inception_resnet_v2 import (
            InceptionResNetV2 as PTModel,
        )

    image_model = PTModel(include_top=False, weights="imagenet")
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    pretrained_image_model = Model(new_input, hidden_layer)
    return pretrained_image_model


def plot_image(image_file):
    """Loads an image from a file and displays it using matplotlib.

    This is a utility function for quick visualization of images.

    Args:
        image_file: A string path to the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    plt.imshow(image)
    plt.show()
