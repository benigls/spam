#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from nltk import tokenize
from nltk.corpus import stopwords

from spam.common.exception import IllegalArgumentError


class Preprocess:
    """ Preprocess class. """
    def __init__(self, **kwargs):
        self.dataset = kwargs.pop('dataset', None)
        self.max_len = kwargs.pop('max_len', 5000)
        self.max_words = kwargs.pop('max_words', 800)
        self.mode = kwargs.pop('mode', 'tfidf')
        self.classes = kwargs.pop('classes', 2)
        self.read_csv = kwargs.pop('read_csv', False)
        self.read_csv_filepath = kwargs.pop('read_csv_filepath', None)

        if self.read_csv:
            self.dataset = pd.read_csv(self.read_csv_filepath)

        for key, item in kwargs.items():
            raise IllegalArgumentError(
                'Keyword argument {} with a value of  {}, '
                'doesn\'t recognize.'.format(key, item))

    def _clean(self, text):
        """ Remove words with non alphanumeric characters and
        remove stopwords in text.
        """
        return ' '.join([w for w in tokenize.word_tokenize(text)
                         if w.isalnum()
                         if w not in stopwords.words('english')])

    def clean_data(self):
        """ Clean data. """
        self.dataset['body'] = \
            self.dataset['body'].apply(lambda text: self._clean(text))

        self.dataset['body'].replace('', np.nan, inplace=True)

        self.dataset = self.dataset.dropna()

    def get_feature_matrix(self, x=None):
        """ Generate feature matrix. """
        clean = lambda words: [str(word)
                               for word in words
                               if type(word) is not float]

        x_unlabel = clean(x[0])
        x_train = clean(x[1])
        x_test = clean(x[2])

        tokenizer = Tokenizer(nb_words=self.max_words)
        tokenizer.fit_on_texts(x_unlabel)

        # save the list of words in the vocabulary
        self.vocabulary = tokenizer.word_counts

        X_unlabel = tokenizer.texts_to_matrix(x_unlabel,
                                              mode=self.mode)
        X_unlabel = pad_sequences(X_unlabel, maxlen=self.max_len,
                                  dtype='float64')

        X_train = tokenizer.texts_to_matrix(x_train, mode=self.mode)
        X_train = pad_sequences(X_train, maxlen=self.max_len, dtype='float64')

        X_test = tokenizer.texts_to_matrix(x_test, mode=self.mode)
        X_test = pad_sequences(X_test, maxlen=self.max_len, dtype='float64')

        return X_unlabel, X_train, X_test

    def get_label_vector(self, y=None):
        """ Preprocess y. """
        y_train = y[0]
        y_test = y[1]

        y_train = np.asarray(y_train, dtype='int32')
        y_test = np.asarray(y_test, dtype='int32')

        Y_train = np_utils.to_categorical(y_train, self.classes)
        Y_test = np_utils.to_categorical(y_test, self.classes)

        return (y_train, Y_train), (y_test, Y_test)
