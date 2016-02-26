#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import re
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
from nltk.corpus import stopwords


def tokenizer(text):
    """ A function that splits a text. """
    return tokenize.word_tokenize(text)


def regex(text):
    """ Remove all words except alphanumeric characters and
    remove the `Subject:`
    """
    clean_text = re.sub('Subject:', '', text)
    return ' '.join([w for w in tokenizer(clean_text) if w.isalnum()])


def remove_stopwords(word_list):
    """ A function that remove stopwords from a list of words
    and lemmatize it and remove mispelled words.
    """
    return [word for word in word_list
            if word not in stopwords.words('english')]


def clean_text(subject, body):
    """ A function that cleans text (regex, token, stop). """
    subject_list = remove_stopwords(tokenizer(regex(subject)))
    body_list = remove_stopwords(tokenizer(regex(body)))
    return ' '.join(subject_list), ' '.join(body_list)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(success=0, fail=0)
def read_email(path, clean=True):
    """ A function that accepts file paths and return it's contents.
    """
    with open(path, 'r', encoding='iso-8859-1') as file:
        try:
            content = file.readlines()
            subject = content.pop(0)
            body = ''.join(content)

            read_email.success += 1
        except UnicodeDecodeError:
            content = ''
            read_email.fail += 1

        file.close()

    sys.stdout.write('\rSuccess: {} \t Fail: {}'.format(
        read_email.success, read_email.fail
    ))
    sys.stdout.flush()

    if clean:
        subject, body = clean_text(subject, body)

    return subject, body


def feature_matrix(dataset=None, max_words=5000, max_len=800, mode='tfidf'):
    """ Transforms panda series to count matrix and normalize it. """
    clean = lambda words: [str(word)
                           for word in words
                           if type(word) is not float]

    x_unlabel = clean(dataset[0])
    x_train = clean(dataset[1])
    x_test = clean(dataset[2])

    tokenizer = Tokenizer(nb_words=max_words)
    tokenizer.fit_on_texts(x_unlabel)

    X_unlabel = tokenizer.texts_to_matrix(x_unlabel, mode=mode)
    X_unlabel = pad_sequences(X_unlabel, maxlen=max_len, dtype='float64')

    X_train = tokenizer.texts_to_matrix(x_train, mode=mode)
    X_train = pad_sequences(X_train, maxlen=max_len, dtype='float64')

    X_test = tokenizer.texts_to_matrix(x_test, mode=mode)
    X_test = pad_sequences(X_test, maxlen=max_len, dtype='float64')

    return X_unlabel, X_train, X_test, tokenizer.word_count
