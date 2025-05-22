"""Provides functions for running inference with the image captioning model.

This module includes functions to generate a caption for a given image
(test_image) and to visualize the attention mechanism (plot_attention),
showing how the model focuses on different image regions while generating words.
It also contains an example of how to use these functions when the script is
executed directly.
"""

import os  # Added for os.path.exists and os.remove
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image  # Assuming PIL is used for Image.open


def test_image(
    image_path,
    image_features_extract_model,
    encoder,
    decoder,
    tokenizer,
    max_length,
    attention_features_shape,
):
    """Generates a caption for a single image using the trained model.

    The process involves:
    1. Loading and preprocessing the image using `load_image` (assumed to be defined elsewhere).
    2. Extracting image features using the `image_features_extract_model` (CNN).
    3. Passing features through the `encoder`.
    4. Iteratively decoding to generate caption tokens:
        - Starting with the '<start>' token.
        - Feeding the current token and encoder output to the `decoder`.
        - Receiving predictions, new decoder state, and attention weights.
        - Storing attention weights for later visualization.
        - Selecting the token with the highest probability as the next input.
    5. Appending predicted words to the result list until '<end>' token or `max_length`.

    Args:
        image_path (str): Path to the input image file.
        image_features_extract_model (tf.keras.Model): The pre-trained CNN model
                                                      used for feature extraction.
        encoder (tf.keras.Model): The encoder part of the captioning model.
        decoder (tf.keras.Model): The decoder part of the captioning model.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer object
                                                           used for word-to-index
                                                           and index-to-word mapping.
        max_length (int): The maximum length of the caption to generate.
        attention_features_shape (int): The number of attention features, used to
                                        initialize the attention plot.

    Returns:
        tuple:
            - result (list): A list of strings representing the generated caption tokens.
            - attention_plot (np.ndarray): A numpy array storing the attention weights
                                           for each generated token. Shape: (len(result), attention_features_shape).
    """
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)

    # load_image function is not defined in the snippet, assuming it exists
    # and returns a preprocessed image tensor and the image path.
    # For this refactoring, let's assume load_image handles preprocessing needed for image_features_extract_model
    # The following line was redundant and used undefined 'image'
    # temp_input = tf.expand_dims(load_image(image)[0], 0)
    # Corrected: temp_input is already defined using image_path few lines above.
    # This was an error from a previous refactoring step. The first temp_input definition is the correct one.
    # The following lines correctly use the *first* temp_input.
    # However, the original code had temp_input defined once, then used image_features_extract_model with it.
    # The error was in the line "temp_input = tf.expand_dims(load_image(image)[0], 0)" which was an artifact.
    # The actual flow should be:
    # 1. load_image(image_path) -> get tensor
    # 2. expand_dims -> temp_input
    # 3. image_features_extract_model(temp_input) -> img_tensor_val
    # The code structure was a bit confusing due to a duplicated line.
    # Let's ensure the first temp_input is used for img_tensor_val.

    # The first `temp_input` definition is:
    # temp_input = tf.expand_dims(load_image(image_path)[0], 0)
    # This is correct and uses image_path.
    # The line `temp_input = tf.expand_dims(load_image(image)[0], 0)` was the error.
    # I will remove the erroneous line. The existing code below it is fine.

    img_tensor_val = image_features_extract_model(
        temp_input
    )  # This temp_input is from load_image(image_path)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])
    )

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == "<end>":
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[: len(result), :]
    return result, attention_plot


def plot_attention(image_path, result, attention_plot):
    """Visualizes the attention weights on the image for each generated caption word.

    This function displays the input image and overlays a heatmap of attention
    weights for each word in the generated caption. This helps to understand
    which parts of the image the model focused on when predicting a particular word.

    Args:
        image_path (str): Path to the original image file.
        result (list): A list of strings representing the generated caption tokens.
        attention_plot (np.ndarray): A numpy array of attention weights, where
                                     attention_plot[i] corresponds to the attention
                                     weights for result[i]. Expected shape is
                                     (len(result), N), where N can be reshaped to
                                     a 2D attention map (e.g., 8x8).
    """
    temp_image = np.array(Image.open(image_path))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap="gray", alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# Helper function (assumed to exist based on original snippet)
def load_image(image_path):
    """Loads and preprocesses an image for the feature extraction model.
    This is a placeholder and needs to be implemented based on the actual
    preprocessing steps used during training (e.g., resizing, normalization).
    """
    # Example implementation:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))  # Example resize, adjust as per model
    img = tf.keras.applications.inception_v3.preprocess_input(img)  # Example preprocess
    return img, image_path


def run_inference_example(
    img_name_val,
    cap_val,
    image_features_extract_model,
    encoder,
    decoder,
    tokenizer,
    max_length,
    attention_features_shape,
):
    """Runs a demonstration of the captioning and attention plotting.

    Selects a random image from the validation set, generates a caption,
    prints the real and predicted captions, and plots the attention.

    Args:
        img_name_val (list): List of image paths from the validation set.
        cap_val (list): List of corresponding tokenized ground truth captions.
        image_features_extract_model (tf.keras.Model): Pre-trained CNN.
        encoder (tf.keras.Model): Captioning model's encoder.
        decoder (tf.keras.Model): Captioning model's decoder.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer.
        max_length (int): Max caption length.
        attention_features_shape (int): Shape of attention features.
    """
    if not img_name_val:
        print("Validation image list (img_name_val) is empty. Cannot run example.")
        return

    rid = np.random.randint(0, len(img_name_val))
    image_path = img_name_val[rid]
    real_caption_tokens = [
        tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]
    ]  # Assuming 0 is padding
    real_caption = " ".join(real_caption_tokens)

    # The original script called `evaluate(image)`.
    # Assuming `test_image` is the intended function for single image captioning.
    result, attention_plot = test_image(
        image_path,
        image_features_extract_model,
        encoder,
        decoder,
        tokenizer,
        max_length,
        attention_features_shape,
    )

    print("Real Caption:", real_caption)
    print("Prediction Caption:", " ".join(result))
    plot_attention(image_path, result, attention_plot)

    # Display the image using PIL to match original script's behavior
    try:
        Image.open(image_path).show()
    except Exception as e:
        print(f"Error opening image with PIL: {e}")


if __name__ == "__main__":
    # This block demonstrates how to run the inference example.
    # In a real application, you would load your trained models, tokenizer,
    # configuration, and data here.

    # --- Placeholder: Load your model, tokenizer, config, and data ---
    # Example:
    # config = load_config() # Implement this
    # vocabulary = load_vocabulary(config.vocabulary_file) # Implement this
    # tokenizer = vocabulary.tokenizer
    # model = build_model(config, vocabulary) # Implement this
    # encoder = model.encoder
    # decoder = model.decoder
    # image_features_extract_model = get_image_encoder(config.cnn) # from images.py
    # max_length = config.max_length
    # attention_features_shape = model.decoder.attention.units # Or however it's defined

    # _, _, coco_eval = prepare_eval_data(config) # from dataset.py to get val names
    # img_name_val = [os.path.join(config.eval_image_dir, coco_eval.imgs[id]['file_name']) for id in coco_eval.getImgIds()]
    # cap_val = ... # load corresponding captions if needed for "real_caption"

    # For demonstration purposes, using dummy/placeholder values:
    print("Running __main__ block in inference.py with placeholder values.")
    print("To run a full example, you need to load actual models, data, and config.")

    # Dummy Tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.word_index = {
                "<start>": 1,
                "<end>": 2,
                "a": 3,
                "cat": 4,
                "dog": 5,
                ".": 6,
            }
            self.index_word = {v: k for k, v in self.word_index.items()}

    dummy_tokenizer = DummyTokenizer()

    # Dummy Models (these would normally be complex tf.keras.Model instances)
    class DummyModel(tf.keras.Model):
        def __init__(self, name="dummy"):
            super().__init__(name=name)

        def __call__(self, *args, **kwargs):  # Make it callable
            if self.name == "image_features_extract_model":
                # (batch_size, height, width, channels) -> (batch_size, new_height, new_width, features)
                return tf.random.uniform(
                    shape=(args[0].shape[0], 8, 8, 2048)
                )  # Example output shape
            elif self.name == "encoder":
                # (batch_size, num_features, feature_depth) -> (batch_size, num_features, units)
                return tf.random.uniform(
                    shape=(args[0].shape[0], args[0].shape[1], 512)
                )  # Example output shape
            return tf.random.uniform(shape=(1, 1))

        def reset_state(self, batch_size):  # For decoder
            return tf.zeros((batch_size, 512))  # Example state shape

    class DummyDecoder(tf.keras.Model):
        def __init__(self, name="dummy_decoder"):
            super().__init__(name=name)

        def __call__(self, dec_input, features, hidden_state):
            # dec_input: (batch_size, 1)
            # features: (batch_size, num_features, units)
            # hidden_state: (batch_size, rnn_units)
            # returns: predictions (batch_size, vocab_size), new_hidden_state, attention_weights (batch_size, num_features)
            vocab_size = len(dummy_tokenizer.word_index) + 1
            num_features = features.shape[1]
            predictions = tf.random.uniform(shape=(dec_input.shape[0], vocab_size))
            new_hidden_state = tf.random.uniform(shape=hidden_state.shape)
            attention_weights = tf.random.uniform(
                shape=(dec_input.shape[0], num_features)
            )
            return predictions, new_hidden_state, attention_weights

        def reset_state(self, batch_size):
            return tf.zeros((batch_size, 512))

    dummy_image_features_extract_model = DummyModel(name="image_features_extract_model")
    dummy_encoder = DummyModel(name="encoder")
    dummy_decoder = DummyDecoder()  # More specific dummy for decoder

    dummy_max_length = 5
    dummy_attention_features_shape = 64  # Should match features.shape[1] from encoder output if attention is over that

    # Dummy image path (replace with a real image path if you want to test Image.open)
    # Create a dummy image file for the example to run without error
    try:
        from PIL import Image as PILImage

        dummy_img = PILImage.new("RGB", (100, 100), color="red")
        dummy_image_path = "dummy_image.jpg"
        dummy_img.save(dummy_image_path)
        dummy_img_name_val = [dummy_image_path]
        dummy_cap_val = [[1, 3, 4, 2]]  # Corresponds to <start> a cat <end>
    except ImportError:
        print(
            "Pillow library is not installed. Cannot create dummy image for the example."
        )
        print("Please install Pillow: pip install Pillow")
        dummy_img_name_val = []
        dummy_cap_val = []
    except Exception as e:
        print(f"Error creating dummy image: {e}")
        dummy_img_name_val = []
        dummy_cap_val = []

    if dummy_img_name_val:
        run_inference_example(
            img_name_val=dummy_img_name_val,
            cap_val=dummy_cap_val,
            image_features_extract_model=dummy_image_features_extract_model,
            encoder=dummy_encoder,
            decoder=dummy_decoder,
            tokenizer=dummy_tokenizer,
            max_length=dummy_max_length,
            attention_features_shape=dummy_attention_features_shape,
        )
        # Clean up dummy image
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
    else:
        print("Skipping run_inference_example due to missing dummy data or Pillow.")
