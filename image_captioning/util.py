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

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

def plot_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    plt.imshow(image)
    plt.show()
