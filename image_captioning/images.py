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

IMAGE_SIZE = {'vgg16': (224,224),
              'inception_v3': (299,299),
              'xception': (299,299),
              'resnet50': (224,224),
              'nasnet_large': (331,331),
              'inception_resnet_v2': (299,299)}

def preprocess_images(config):
    """Extract image features and save them as numpy arrays.
    
    This will process both the training and the validation datasets.
    
    Arguments:
        config (util.Config): Values for various configuration options.
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

    # Obtain image files for training dataset
    coco_train = COCO(config.train_captions_file)
    train_image_ids = coco_train.get_unique_image_ids()
    # train_image_filenames = coco_train.get_image_filenames(train_image_ids)
    
    # Obtain image files for evaluation dataset
    coco_eval = COCO(config.eval_captions_file)
    eval_image_ids = coco_eval.get_unique_image_ids()
    # eval_image_filenames = coco_eval.get_image_filenames(eval_image_ids)
    
    image_files = coco_train.get_image_files(config.train_image_dir, train_image_ids) \
                + coco_eval.get_image_files(config.eval_image_dir, eval_image_ids)

    # Create a dataset with images ready to be fed into the encoder (cnn for extracting image features) in batches 
    image_dataset = Dataset.from_tensor_slices(sorted(set(image_files)))
    image_dataset = image_dataset.map(
            image_preprocessing_function(cnn), num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).batch(batchsize)

    features_dir = os.path.join(features_dir, cnn)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    for images, image_filenames in tqdm(image_dataset, desc='image_batch'):
        # obtain batch of image features
        image_features = encoder(images)
        image_features = tf.reshape(
            image_features, (image_features.shape[0], -1, image_features.shape[3]))
        # Save image features as '.npy' files
        for img_features, img_filename in zip(image_features, image_filenames):
            features_path = os.path.join(
                    config.image_features_dir, cnn, 
                    os.path.basename(img_filename.numpy().decode('utf-8'))
                    )
            np.save(features_path, img_features.numpy())

def image_preprocessing_function(cnn):
    """Returns a function that can load and preprocess images for a specific cnn model.
    
    Args:
        cnn {string} -- Name of the CNN used to encode image features
    
    Returns:
        function -- Function to load and prepare images for the given CNN
    """
    image_size = IMAGE_SIZE[cnn]
    if cnn == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif cnn == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif cnn == 'xception':
        from tensorflow.keras.applications.xception import preprocess_input
    elif cnn == 'nasnet_large':
        from tensorflow.keras.applications.nasnet import preprocess_input
    elif cnn == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif cnn == 'inception_resnet_v2':
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

    def load_and_preprocess_image(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = preprocess_input(image)
        return image, image_file
        
    return load_and_preprocess_image

def get_image_encoder(cnn):
    """Create feature extraction layer for the specified cnn.
    
    Supports Inception_V3 and NASNet
    """
    if cnn == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16 as PTModel
    elif cnn == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3 as PTModel
    elif cnn == 'xception':
        from tensorflow.keras.applications.xception import Xception as PTModel
    elif cnn == 'nasnet_large':
        from tensorflow.keras.applications.nasnet import NASNetLarge as PTModel
    elif cnn == 'resnet50':
        from tensorflow.keras.applications.resnet50 import ResNet50 as PTModel
    elif cnn == 'inception_resnet_v2':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel

    image_model = PTModel(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    pretrained_image_model = Model(new_input, hidden_layer)
    return pretrained_image_model


def plot_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    plt.imshow(image)
    plt.show()