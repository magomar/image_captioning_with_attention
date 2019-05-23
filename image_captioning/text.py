import json
import os
import pickle

import numpy as np
import tensorflow as tf

from absl import logging
from cocoapi.pycocotools.coco import COCO
from images import plot_image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

class Vocabulary(object):
    """Helper class used to process captions
    
    """

    def __init__(self, vocabulary_size, sentence_length=None, save_file=None):
        """[summary]
        
        Arguments:
            vocabulary_size {integer} -- Vocabulary size (max number of words)
        
        Keyword Arguments:
            save_file {String} -- File path to serialize vocabulary (default: {None})
        """
        self.size = vocabulary_size
        self.sentence_length = sentence_length
        if save_file is not None:
            self.load(save_file)

    def build(self, sentences):
        """Builds a vocabulary for the provided sentences.
        
        A vocabulary is represented by a Tokenizer object which is fitted to the
        sentences, but limitted to a maximum number of words (the `size` of the vocabulary),
        using frequency as the criteria to filter words

        Arguments:
            sentences {list} -- A list of sentences

        """
        # choosing the top k words from the vocabulary
        self.tokenizer = Tokenizer(
                                num_words=self.size,
                                oov_token="<unk>",
                                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(sentences)
        # adds a padding token
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

    def process_sentence(self, sentence):
        """Tokenizes and pads a single sentence
        
        """
        return process_sentences([sentence])[0]

    def process_sentences(self, sentences):
        """Tokenize and pads a single sentence
        
        Args:
            sentence (String): A text sentence
        
        Returns:
            sequences: tokenized & padded captions
        """

        # Tokenize sentences (convert them to sequences of indexes)
        sequences = self.tokenizer.texts_to_sequences(text_captions)
        max_length = max(len(seq) for seq in sequences)
        logging.info("Max caption length: %d", max_length)
        # Pad sequences to met the max length.
        # If maxlen parameter is not provided, it is computed automatically
        sequences = pad_sequences(
            sequences, maxlen=max_length, padding='post')
        return sequences

    def save(self, save_file):
        """ Save the vocabulary to a file.
        """
        with open(save_file, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, save_file):
        """ Load the vocabulary from a file. 
        
        """
        with open(save_file, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file.
    
    """
    coco = COCO(config.train_captions_file)
    text_captions = coco.get_text_captions()
    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.build(text_captions)
    vocabulary.save(config.vocabulary_file)
    return vocabulary