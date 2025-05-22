"""Provides utility classes and functions for the image captioning project.

Currently, this module contains:
- `ImageHelper`: A class to assist with loading images from a specific path structure.
"""

import os

# import matplotlib # Not strictly needed if only plt is used
# import random # No longer needed as generate_one_sequence_greedy and eval are removed
# matplotlib.use('agg') # This is a backend setting, might be needed if running in non-GUI environment

# import matplotlib.pyplot as plt  # Removed as it's unused in this module
import tensorflow as tf


class ImageHelper(object):
    """Helper class to load images based on a base path and filename prefix convention.

    This class simplifies the process of constructing image file paths and loading
    image data, assuming images are stored with a common prefix and a zero-padded
    ID in their filenames (e.g., 'COCO_train2014_000000000001.jpg').
    """

    def __init__(self, image_path, prefix):
        """Initializes the ImageHelper.

        Args:
            image_path (str): The absolute or relative base directory path
                              where images are stored.
            prefix (str): The common prefix for image filenames (e.g.,
                          'COCO_train2014_').
        """
        self.path = os.path.abspath(image_path)
        self.prefix = prefix

    def load_image(self, image_id):
        """Loads and decodes an image given its ID.

        Constructs the full image file path using `get_image_file`, reads the
        file content, and decodes it as a JPEG image with 3 color channels.

        Args:
            image_id (int): The integer ID of the image to load. This ID will
                            be zero-padded to 12 digits in the filename.

        Returns:
            tf.Tensor: A TensorFlow tensor representing the decoded image.
                       Shape: (height, width, 3).
        """
        image_file_path = self.get_image_file(image_id)
        image = tf.io.read_file(image_file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def get_image_file(self, image_id):
        """Constructs the full file path for an image given its ID.

        The filename is constructed as: `self.path / (self.prefix + 12-digit-zero-padded-image_id + .jpg)`.
        For example, if `image_path` is '/data/images', `prefix` is 'train_',
        and `image_id` is 123, the path would be '/data/images/train_000000000123.jpg'.

        Args:
            image_id (int): The integer ID of the image.

        Returns:
            str: The fully constructed image file path.
        """
        image_file = os.path.join(self.path, "%s%012d.jpg" % (self.prefix, image_id))
        return image_file
