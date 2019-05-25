import os
import sys

import tensorflow as tf

from absl import app, flags, logging
from config import Config
from images import preprocess_images
from evaluation import evaluate
from text import load_or_build_vocabulary
from training import train

FLAGS = flags.FLAGS
flags.DEFINE_string('phase', None,
                    'The phase can be prepare, train, eval or infer')

flags.DEFINE_boolean('load', True,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

# flags.DEFINE_string('model_file', None,
#                     'If specified, load a pretrained model from this file')

flags.DEFINE_integer('epochs', None,
                    'If specified, train for this number of epochs. It will override config options')

flags.DEFINE_integer('examples', None,
                    'If specified, restrict the number of training examples to this number. It will override config options')

# Required flags
flags.mark_flag_as_required("phase")

# Logging
logging.set_verbosity(logging.INFO)
tf.get_logger().setLevel('ERROR')

def main(argv):
    """python image_captioning/main.py --train --log_dir log
    
    """

    del argv  # Unused.
    config = Config()
    config.phase = FLAGS.phase

    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
        file=sys.stderr)

    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        log_file='absl_logging'
        logging.get_absl_handler().use_absl_log_file(log_file, FLAGS.log_dir)

    if FLAGS.epochs:
        config.num_epochs=FLAGS.epochs

    if FLAGS.examples:
        config.num_train_examples=FLAGS.examples

    if FLAGS.load is False:
        config.resume_from_checkpoint = False 

    logging.info('Running %s phase', config.phase)

    if FLAGS.phase == 'prepare':
        # build vocabulary
        load_or_build_vocabulary(config)
        # extracts and saves image features for later use
        if config.extract_image_features:
            preprocess_images(config)

    elif FLAGS.phase == 'train': 
        # training phase: build and trains an image captioning model
        train(config)
        
    elif FLAGS.phase == 'eval':
        # evaluation phase: evaluates a saved model on the validation data
        evaluate(config)

if __name__ == '__main__':
  app.run(main)