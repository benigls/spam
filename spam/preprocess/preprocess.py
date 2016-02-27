#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
from nltk.corpus import stopwords

from spam.common.exception import IllegalArgumentError


class Preprocess:
    """ Preprocess class. """
    def init(self, **kwargs):
        self.dataset = kwargs.pop('dataset', None)
        self.max_len = kwargs.pop('max_len', 5000)
        self.max_words = kwargs.pop('max_words', 800)
        self.mode = kwargs.pop('mode', 'tfidf')
        self.csv_filename = kwargs.pop('csv_filename', None)
        self.read_csv = kwargs.pop('read_csv', False)
        self.vocabulary = kwargs.pop('vocabulary', None)

        for key, item in kwargs.items():
            raise IllegalArgumentError(
                'Keyword argument {} with a value of  {}, '
                'doesn\'t recognize.'.format(key, item))

    def clean(self, text):
        """ Remove words with non alphanumeric characters and
        remove stopwords in text.
        """
        return ' '.join([w for w in tokenize.word_tokenize(text)
                         if w.isalnum()
                         if w not in stopwords.words('english')])

    def clean_data(path, clean=True):
        """ Clean data. """
        pass

    def feature_matrix(self):
        """ Generate feature matrix. """
        clean = lambda words: [str(word)
                               for word in words
                               if type(word) is not float]

        x_unlabel = clean(self.dataset[0])
        x_train = clean(self.dataset[1])
        x_test = clean(self.dataset[2])

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

    def to_csv(self, path=None, name=None):
        """ Export dataset into csv file. """
        self.dataset.to_csv('{}/{}.csv'.format(path, name))
