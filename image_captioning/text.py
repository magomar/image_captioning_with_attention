"""Handles text processing for image captioning.

This module provides the `Vocabulary` class, which is responsible for:
- Building a vocabulary from a collection of sentences.
- Tokenizing sentences into sequences of integers.
- Padding sequences to a uniform length.
- Converting sequences back to text.
- Saving and loading the vocabulary.

It also includes helper functions like `load_or_build_vocabulary`
to manage vocabulary persistence and `max_sequence_length` to find the
longest sequence in a list.
"""

import os
import pickle

from absl import logging
from cocoapi.pycocotools.coco import COCO
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Vocabulary(object):
    """Helper class used to process captions"""

    def __init__(self, vocabulary_size, sequence_length=None, save_file=None):
        """Initializes a Vocabulary, optionally loading from `save_file`.

        Args:
            vocabulary_size: Max words in vocab (by frequency).
            sequence_length: Fixed length for padding token sequences. If None,
                             it may be set later (e.g., from max length in a
                             dataset). Defaults to None.
            save_file: Path to a saved vocabulary (tokenizer) file. If given,
                       vocabulary is loaded. Defaults to None.
        """
        self.size = vocabulary_size
        self.sequence_length = sequence_length
        if save_file is not None:
            self.load(save_file)

    def build(self, sentences):
        """Builds the vocabulary from a list of sentences.

        Uses `tf.keras.preprocessing.text.Tokenizer` to map words to indices,
        limited to `self.size` most frequent words. Handles out-of-vocabulary
        words with `<unk>` and adds a `<pad>` token. `<start>` and `<end>`
        tokens should be in `sentences` to be indexed.

        Args:
            sentences: A list of sentences (strings) for vocabulary building.
        """
        self.tokenizer = Tokenizer(
            num_words=self.size,
            oov_token="<unk>",
            filters='!"#$%&()*+.,-/:;=?@[]\\^_`{|}~ ',  # Corrected for W605
        )
        self.tokenizer.fit_on_texts(sentences)
        # Add padding token.
        self.tokenizer.word_index["<pad>"] = 0
        self.tokenizer.index_word[0] = "<pad>"
        self.setup()

    def setup(self):
        """Finalizes vocabulary setup after tokenizer is created/loaded.

        Adjusts `self.size` if actual unique words are fewer than specified.
        Sets attributes for special token indices (`<start>`, `<end>`, `<pad>`),
        assuming `<start>` and `<end>` were in the data if needed.
        """
        # +1 because index 0 is reserved for padding.
        num_words = len(self.tokenizer.word_index) + 1
        if self.size > num_words:
            self.size = num_words

        # Fallback to <unk> if <start> or <end> were not in training sentences.
        self.start = self.tokenizer.word_index.get(
            "<start>", self.tokenizer.word_index.get("<unk>")
        )
        self.end = self.tokenizer.word_index.get(
            "<end>", self.tokenizer.word_index.get("<unk>")
        )
        self.pad = self.tokenizer.word_index.get("<pad>", 0)  # Should be 0

    def seq2text(self, sequence):
        """Converts a sequence of token IDs back to a sentence string.

        Conversion stops at `<end>` or `<pad>` tokens.

        Args:
            sequence: A list of token IDs.

        Returns:
            The reconstructed sentence string.
        """
        sentence = ""
        for idx in sequence:
            if idx == self.pad or idx == self.end:
                break
            # Ensure index is valid before accessing index_word
            if idx in self.tokenizer.index_word:
                sentence += " " + self.tokenizer.index_word[idx]
        return sentence.lstrip()

    def process_sentences(self, sentences):
        """Converts sentences to tokenized and padded sequences.

        Sentences are tokenized to integer sequences, then padded (post-padding)
        to `self.sequence_length`. If `self.sequence_length` is None, it's set
        by the longest sequence in the current batch.

        Args:
            sentences: A list of sentences to process.

        Returns:
            A NumPy array of tokenized and padded sequences, shape:
            (num_sentences, sequence_length).
        """
        sequences = self.tokenizer.texts_to_sequences(sentences)
        current_max_length = max_sequence_length(sequences) if sequences else 0
        logging.info(
            f"Processed {len(sequences)} sentences. "
            f"Max length in current batch = {current_max_length}"
        )

        pad_to_length = self.sequence_length
        if pad_to_length is None:
            pad_to_length = current_max_length
            logging.info(
                f"Using max_length from current batch for padding: {pad_to_length}"
            )
        else:
            logging.info(f"Using fixed sequence_length for padding: {pad_to_length}")

        sequences = pad_sequences(sequences, maxlen=pad_to_length, padding="post")
        return sequences

    def save(self, save_file):
        """Saves the Keras Tokenizer to a file using pickle.

        Args:
            save_file: Path to save the tokenizer.
        """
        with open(save_file, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Vocabulary (tokenizer) saved to {save_file}")

    def load(self, save_file):
        """Loads Keras Tokenizer from a file and calls `setup()`.

        Args:
            save_file: Path to load the tokenizer from.
        """
        with open(save_file, "rb") as handle:
            self.tokenizer = pickle.load(handle)
        self.setup()
        logging.info(f"Vocabulary (tokenizer) loaded from {save_file}")


def load_or_build_vocabulary(config, sentences=None):
    """Loads vocabulary from file or builds it if non-existent.

    Checks for `config.vocabulary_file`. If it exists, loads it.
    Otherwise, if `sentences` are provided (or loadable via `config`),
    builds, saves, and returns a new vocabulary.

    Args:
        config: Configuration object with `vocabulary_size`, `max_length` (for
                sequence_length), `vocabulary_file` (save/load path), and
                `train_captions_file` (if sentences need loading).
        sentences: List of sentences to build vocab from if not existing.
                   If None, uses `config.train_captions_file`. Defaults to None.

    Returns:
        The loaded or newly built `Vocabulary` object.
    """
    seq_len = config.max_length if config.limit_length else None
    vocabulary = Vocabulary(config.vocabulary_size, sequence_length=seq_len)

    if not os.path.exists(config.vocabulary_file):
        logging.info(f"Vocabulary file {config.vocabulary_file} not found.")
        if sentences is None:
            if not config.train_captions_file or not os.path.exists(
                config.train_captions_file
            ):
                raise FileNotFoundError(
                    f"Captions file not found: {config.train_captions_file}, "
                    "and no sentences provided for vocab building."
                )
            logging.info(
                f"Loading sentences from {config.train_captions_file} "
                "to build vocabulary."
            )
            coco = COCO(config.train_captions_file)
            sentences = coco.get_all_captions()

        logging.info("Building the vocabulary...")
        vocabulary.build(sentences)
        vocabulary.save(config.vocabulary_file)
    else:
        logging.info(f"Loading vocabulary from {config.vocabulary_file}.")
        vocabulary.load(config.vocabulary_file)

    logging.info(
        "Effective vocabulary size (unique words + special tokens): "
        f"{len(vocabulary.tokenizer.word_index) + 1}"
    )
    logging.info(f"Target vocabulary size (max words from config): {vocabulary.size}")
    return vocabulary


def max_sequence_length(sequences):
    """Calculates the max length of sequences in a list of sequences.

    Args:
        sequences: A list where each inner list is a sequence (e.g., token IDs).

    Returns:
        Max length of sequences. Returns 0 if input is empty or contains only
        empty sequences.
    """
    if not sequences:
        return 0
    # Ensure seq is not None or empty before calling len(seq)
    return max(len(seq) for seq in sequences if seq)
