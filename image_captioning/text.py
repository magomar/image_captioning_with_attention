import json
import os
import pickle

import numpy as np
import tensorflow as tf

from absl import logging
from cocoapi.pycocotools.coco import COCO
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
            sentences {string list} -- A list of sentences

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
        self.setup()


    def setup(self):
        num_words = len(self.tokenizer.word_index) + 1
        if self.size > num_words:
            self.size = num_words
        self.start = self.tokenizer.word_index['<start>']
        self.end = self.tokenizer.word_index['<end>']
        self.pad = self.tokenizer.word_index['<pad>']

    def sequence2sentence(self, sequence):
        """Converts a sequence back to a sentence
        
        """
        # sentence = ' '.join([self.tokenizer.index_word[i] for i in sequence if i not in [self.pad,self.end]])
        sentence = ''
        for idx in sequence:
            if idx in [self.pad, self.end]:
                break
            else:
                sentence += ' ' + self.tokenizer.index_word[idx]
        return sentence.lstrip()

    def process_sentences(self, sentences):
        """Tokenize and pads a list of sentences
        
        Arguments:
            sentences (string list): A list of sentences
        
        Returns:
            sequences: tokenized & padded captions
        """

        # Tokenize sentences (convert them to sequences of indexes)
        sequences = self.tokenizer.texts_to_sequences(sentences)
        max_length = max_sequence_length(sequences)
        logging.info("Processed %d sentences. Max length = %d", len(sequences), max_length)
        if self.sentence_length is None:
            self.sentence_length = max_length
        else:
            logging.info("But it is set to : %d", self.sentence_length)
        # Pad sequences to met the max length.
        # If maxlen parameter is not provided, it is computed automatically
        sequences = pad_sequences(sequences, maxlen=self.sentence_length, padding='post')
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
        self.setup()

def load_or_build_vocabulary(config, sentences = None):
    vocabulary = Vocabulary(config.vocabulary_size)
    if not os.path.exists(config.vocabulary_file):
        if sentences is None:
            coco = COCO(config.train_captions_file)
            sentences = coco.get_all_captions()
        logging.info("Building the vocabulary...")
        vocabulary.build(sentences)
        vocabulary.save(config.vocabulary_file)
        logging.info("Vocabulary saved to %s." % config.vocabulary_file)
    else:
        logging.info("Loading vocabulary from %s.", config.vocabulary_file)
        vocabulary.load(config.vocabulary_file)
    logging.info("Num words: %d", len(vocabulary.tokenizer.word_index) + 1)
    logging.info("Vocabulary size = %d" %(vocabulary.size))
    return vocabulary

def max_sequence_length(sequences):
    return max(len(seq) for seq in sequences)