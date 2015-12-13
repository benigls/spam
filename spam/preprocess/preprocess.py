#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import re

from nltk import tokenize
from nltk.corpus import stopwords
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
    A function that remove stopwords from a list of words.
    """
    return [word for word in word_list
            if word not in stopwords.words('english')]


def clean_text(text):
    """
    A function that cleans text (regex, token, stop).
    """
    word_list = remove_stopwords(tokenizer(regex(text)))
    return ' '.join(word_list)


def read_email(path, clean=True):
    """
    A function that accepts file paths and return it's contents.
    """
    with open(path, 'r', encoding='iso-8859-1') as file:
        try:
            content = ''.join(file.readlines())
        except UnicodeDecodeError:
            content = ''

        file.close()

    return clean_text(content) if clean else content


def count_vectorizer(series, test=None):
    """
    A function that transforms panda series to count vectorizer
    If parameter `test` is True the function will only transform the
    series but if the parameter `test` is False the function
    will fit and transform the series.
    """
    fit_transform = True if not test else False
    email_list = [email.encode('utf-8')
                  for email in series.values.tolist()
                  if type(email) is not float]

    vector = CountVectorizer(
        analyzer='word',
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=5000,
    )
    normalizer = Normalizer()

    feat_vector = vector.fit_transform(email_list) \
        if fit_transform else vector.transform(email_list)

    return normalizer.fit_transform(feat_vector).toarray()
