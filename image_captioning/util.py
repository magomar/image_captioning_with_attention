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