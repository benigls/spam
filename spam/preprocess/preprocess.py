#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import re
import sys

from scipy.sparse import csr_matrix
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer


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
    and lemmatize it.
    """
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(word) for word in word_list
            if word not in stopwords.words('english')]


def clean_text(subject, body):
    """
    A function that cleans text (regex, token, stop).
    """
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
    """
    A function that accepts file paths and return it's contents.
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


def count_vectorizer(dataset=None, max_features=2000):
    """ Transforms panda series to count matrix and
    normalize it then transform it to a dense matrix.
    """
    clean = lambda words: [word.encode('utf-8')
                           for word in words
                           if type(word) is not float]

    x_unlabel = clean(dataset[0])
    x_train = clean(dataset[1])
    x_test = clean(dataset[2])

    vector = CountVectorizer(analyzer='word', max_features=max_features)
    normalizer = Normalizer()

    X_unlabel = vector.fit_transform(x_unlabel).astype('float64')
    X_unlabel = csr_matrix(X_unlabel).todense()
    # X_unlabel = normalizer.fit_transform(X_unlabel)

    X_train = vector.transform(x_train).astype('float64')
    X_train = csr_matrix(X_train).todense()
    # X_train = normalizer.transform(X_train)

    X_test = vector.transform(x_test).astype('float64')
    X_test = csr_matrix(X_test).todense()
    # X_test = normalizer.transform(X_test)
    import pdb
    pdb.set_trace()

    return X_unlabel, X_train, X_test
