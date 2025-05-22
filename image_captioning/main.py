"""Main entry point for the image captioning application.

This script handles command-line arguments to orchestrate different phases
of the image captioning pipeline, such as:
- 'prepare': Preprocesses data, builds vocabulary, and extracts image features.
- 'train': Trains the image captioning model.
- 'eval': Evaluates a trained model.
- 'infer': (If implemented) Runs inference on new images.

It uses `absl-py` for flags and application management and loads
configurations from `config.py`.
"""

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
flags.DEFINE_string("phase", None, "The phase can be prepare, train, eval or infer")

flags.DEFINE_boolean(
    "load",
    True,
    (
        "Turn on to load a pretrained model from either the latest checkpoint "
        "or a specified file"
    ),
)

flags.DEFINE_integer(
    "epochs",
    None,
    "If specified, train for this number of epochs. It will override config options",
)

flags.DEFINE_integer(
    "examples",
    None,
    "If specified, restrict the number of training examples to this number. It will override config options",
)

# Required flags
flags.mark_flag_as_required("phase")

# Logging
logging.set_verbosity(logging.INFO)
tf.get_logger().setLevel("ERROR")


def main(argv):
    """Main function to run the image captioning pipeline phases.

    This function serves as the primary entry point after parsing command-line
    flags. It initializes the configuration (`Config`), sets up logging if a
    log directory is specified, and then executes the appropriate pipeline phase
    ('prepare', 'train', 'eval') based on the '--phase' flag.

    It can override parts of the configuration using command-line flags,
    such as the number of epochs or whether to load a pre-trained model.

    Example usage:
        python image_captioning/main.py --phase=train --log_dir=./logs --epochs=20
        python image_captioning/main.py --phase=prepare
        python image_captioning/main.py --phase=eval --load=True

    Args:
        argv: A list of command-line arguments (unused after flag parsing by absl).
    """

    del argv  # Unused.
    config = Config()

    if config.log_dir is not None:
        FLAGS.log_dir = config.log_dir

    config.phase = FLAGS.phase

    print(
        "Running under Python {0[0]}.{0[1]}.{0[2]}".format(sys.version_info),
        file=sys.stderr,
    )

    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        log_file = "absl_logging"
        logging.get_absl_handler().use_absl_log_file(log_file, FLAGS.log_dir)

    if FLAGS.epochs:
        config.num_epochs = FLAGS.epochs

    if FLAGS.load is False:
        config.resume_from_checkpoint = False

    logging.info("Running %s phase", config.phase)

    if FLAGS.phase == "prepare":
        # build vocabulary
        load_or_build_vocabulary(config)
        # extracts and saves image features for later use
        if config.extract_image_features:
            preprocess_images(config)

    elif FLAGS.phase == "train":
        # training phase: build and trains an image captioning model
        train(config)

    elif FLAGS.phase == "eval":
        # evaluation phase: evaluates a saved model on the validation data
        evaluate(config)


if __name__ == "__main__":
    app.run(main)
