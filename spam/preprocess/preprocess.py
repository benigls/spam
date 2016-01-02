#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import re
import sys

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer


def regex(text):
    """
    A function that removes non-alphanumeric, -, _ characters
    and the word `Subject:`, and `re:` in text.
    """
    clean_text = re.sub('Subject:|re:', '', text)
    clean_text = re.sub('[^\w]+', ' ', clean_text)

    return clean_text


def tokenizer(text):
    """
    A function that splits a text.
    """
    return tokenize.word_tokenize(text)


def remove_stopwords(word_list):
    """
    A function that remove stopwords from a list of words
    and lemmatize it.
    """
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(word) for word in word_list
            if word not in stopwords.words('english')]


def clean_text(text):
    """
    A function that cleans text (regex, token, stop).
    """
    word_list = remove_stopwords(tokenizer(regex(text)))
    return ' '.join(word_list)


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
            content = ''.join(file.readlines())
            read_email.success += 1
        except UnicodeDecodeError:
            content = ''
            read_email.fail += 1

        file.close()

    sys.stdout.write('\rSuccess: {} \t Fail: {}'.format(
        read_email.success, read_email.fail
    ))
    sys.stdout.flush()

    return clean_text(content) if clean else content


def count_vectorizer(dataset, max_features=5000):
    """
    A function that transforms panda series to count vectorizer
    If parameter `test` is True the function will only transform the
    series but if the parameter `test` is False the function
    will fit and transform the series.
    """
    clean = lambda x: [email.encode('utf-8')
                       for email in x.values.tolist()
                       if type(email) is not float]

    train_list = clean(dataset[0])
    test_list = clean(dataset[1])

    vector = CountVectorizer(
        analyzer='word',
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=max_features,
    )
    normalizer = Normalizer()

    train_vector = normalizer.fit_transform(
        vector.fit_transform(train_list)
    ).toarray()
    test_vector = normalizer.fit_transform(
        vector.transform(test_list)
    ).toarray() if not dataset else None

    return train_vector, test_vector
