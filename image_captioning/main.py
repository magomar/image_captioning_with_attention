import os
import sys

import tensorflow as tf

from absl import app, flags, logging
from config import Config
from dataset import prepare_train_data, prepare_eval_data
from models import build_model
from training import train

FLAGS = flags.FLAGS
flags.DEFINE_string('phase', None,
                    'The phase can be prepare, train, eval or test')

flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

flags.DEFINE_string('model_file', None,
                    'If specified, load a pretrained model from this file')

# Required flags
flags.mark_flag_as_required("phase")

# Logging

logging.set_verbosity(logging.INFO)
tflogger = tf.get_logger()
tflogger.setLevel('ERROR')

def main(argv):
    """python image_captioning/main.py --log_dir log"""

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

    logging.info('Running %s phase', config.phase)

    if FLAGS.phase == 'prepare':
        # preparation phase (extracts and saves image features for later use)
        config.extract_image_features = True
        train_dataset = prepare_train_data(config)
    elif FLAGS.phase == 'train':
        # training phase
        train_dataset = prepare_train_data(config)
        tokenizer = train_dataset.tokenizer
        model = build_model(tokenizer, config)
        train(model, train_dataset, config)

if __name__ == '__main__':
  app.run(main)