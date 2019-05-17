import os

import matplotlib
# matplotlib.use('agg')

import matplotlib.pyplot as plt
import tensorflow as tf

class ImageHelper(object):
    def __init__(self, image_path, prefix):
        self.path = image_path
        self.prefix = prefix

    def load_image(self, image_id):
        image = tf.io.read_file(get_image_file(image_id))
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def get_image_file(self, image_id):
        image_file = os.path.join(self.path, '%s%012d.jpg' % (self.prefix, image_id))
        return image_file

# def get_coco_train_image_name(image_id):
#     return '/COCO_train2014_' + '%012d.jpg' % (image_id)

# def get_coco_eval_image_name(image_id):
#     return '/COCO_val2014_' + '%012d.jpg' % (image_id)

# def get_image_file(image_path, image_id, prefix):
#     return os.path.join(image_path, '%s%012d.jpg' % (prefix, image_id)

# def load_image(image_path, image_id, prefix):
#         image_file = get_image_file(image_path, image_id, prefix))
#         image = tf.io.read_file(image_file)
#         image = tf.image.decode_jpeg(image, channels=3)
#         return image

def plot_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    plt.imshow(image)
    plt.show()

def load_image_for_preprocessing(image_file):
    """ Loads an image from file, and transforms it into the Inception-V3 format """

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, image_file

def download_coco(config):
    """ Donwload COCO dataset """

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